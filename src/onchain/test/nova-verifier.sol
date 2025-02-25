// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.7.0 <0.9.0;

/*
    Sonobe's Nova + CycleFold decider verifier.
    Joint effort by 0xPARC & PSE.

    More details at https://github.com/privacy-scaling-explorations/sonobe
    Usage and design documentation at https://privacy-scaling-explorations.github.io/sonobe-docs/

    Uses the https://github.com/iden3/snarkjs/blob/master/templates/verifier_groth16.sol.ejs
    Groth16 verifier implementation and a KZG10 Solidity template adapted from
    https://github.com/weijiekoh/libkzg.
    Additionally we implement the NovaDecider contract, which combines the
    Groth16 and KZG10 verifiers to verify the zkSNARK proofs coming from
    Nova+CycleFold folding.
*/


/* =============================== */
/* KZG10 verifier methods */
/**
 * @author  Privacy and Scaling Explorations team - pse.dev
 * @dev     Contains utility functions for ops in BN254; in G_1 mostly.
 * @notice  Forked from https://github.com/weijiekoh/libkzg.
 * Among others, a few of the changes we did on this fork were:
 * - Templating the pragma version
 * - Removing type wrappers and use uints instead
 * - Performing changes on arg types
 * - Update some of the `require` statements 
 * - Use the bn254 scalar field instead of checking for overflow on the babyjub prime
 * - In batch checking, we compute auxiliary polynomials and their commitments at the same time.
 */
contract KZG10Verifier {

    // prime of field F_p over which y^2 = x^3 + 3 is defined
    uint256 public constant BN254_PRIME_FIELD =
        21888242871839275222246405745257275088696311157297823662689037894645226208583;
    uint256 public constant BN254_SCALAR_FIELD =
        21888242871839275222246405745257275088548364400416034343698204186575808495617;

    /**
     * @notice  Performs scalar multiplication in G_1.
     * @param   p  G_1 point to multiply
     * @param   s  Scalar to multiply by
     * @return  r  G_1 point p multiplied by scalar s
     */
    function mulScalar(uint256[2] memory p, uint256 s) internal view returns (uint256[2] memory r) {
        uint256[3] memory input;
        input[0] = p[0];
        input[1] = p[1];
        input[2] = s;
        bool success;
        assembly {
            success := staticcall(sub(gas(), 2000), 7, input, 0x60, r, 0x40)
            switch success
            case 0 { invalid() }
        }
        require(success, "bn254: scalar mul failed");
    }

    /**
     * @notice  Negates a point in G_1.
     * @param   p  G_1 point to negate
     * @return  uint256[2]  G_1 point -p
     */
    function negate(uint256[2] memory p) internal pure returns (uint256[2] memory) {
        if (p[0] == 0 && p[1] == 0) {
            return p;
        }
        return [p[0], BN254_PRIME_FIELD - (p[1] % BN254_PRIME_FIELD)];
    }

    /**
     * @notice  Adds two points in G_1.
     * @param   p1  G_1 point 1
     * @param   p2  G_1 point 2
     * @return  r  G_1 point p1 + p2
     */
    function add(uint256[2] memory p1, uint256[2] memory p2) internal view returns (uint256[2] memory r) {
        bool success;
        uint256[4] memory input = [p1[0], p1[1], p2[0], p2[1]];
        assembly {
            success := staticcall(sub(gas(), 2000), 6, input, 0x80, r, 0x40)
            switch success
            case 0 { invalid() }
        }

        require(success, "bn254: point add failed");
    }

    /**
     * @notice  Computes the pairing check e(p1, p2) * e(p3, p4) == 1
     * @dev     Note that G_2 points a*i + b are encoded as two elements of F_p, (a, b)
     * @param   a_1  G_1 point 1
     * @param   a_2  G_2 point 1
     * @param   b_1  G_1 point 2
     * @param   b_2  G_2 point 2
     * @return  result  true if pairing check is successful
     */
    function pairing(uint256[2] memory a_1, uint256[2][2] memory a_2, uint256[2] memory b_1, uint256[2][2] memory b_2)
        internal
        view
        returns (bool result)
    {
        uint256[12] memory input = [
            a_1[0],
            a_1[1],
            a_2[0][1], // imaginary part first
            a_2[0][0],
            a_2[1][1], // imaginary part first
            a_2[1][0],
            b_1[0],
            b_1[1],
            b_2[0][1], // imaginary part first
            b_2[0][0],
            b_2[1][1], // imaginary part first
            b_2[1][0]
        ];

        uint256[1] memory out;
        bool success;

        assembly {
            success := staticcall(sub(gas(), 2000), 8, input, 0x180, out, 0x20)
            switch success
            case 0 { invalid() }
        }

        require(success, "bn254: pairing failed");

        return out[0] == 1;
    }

    uint256[2] G_1 = [
            0x171149d656ab2678f03a81fb4a13b38cb13c584222498b9f0824377ff4ef1c6c,
            0x1078a9c7358344c97989a825cd493c02502c5979785ab4102b85cab8785e1652
    ];
    uint256[2][2] G_2 = [
        [
            0x2e85a64b176a89f651b755522402780ac5224a690c62c2a3580e2b77391eb85f,
            0x25a630e1b1bb847ca0e32e8d5a2c62d424e4b0d67d8295d094589efba2611485
        ],
        [
            0x105e2254f385a54471f0f072b96fc88fc2a55e98e8fccbdcd5da2729f42941c7,
            0x1478cea7a3717eed37243042378ca4d61c6d69d433ecbf2967d813cb1adc02ae
        ]
    ];
    uint256[2][2] VK = [
        [
            0x213df544b48e424ce1eca450ed03a5496169eddd748b653832354ca0bcf23f62,
            0x2959a960103c0295e1a16715e806239ff6c59e4c0743824c117de81ae7f85c96
        ],
        [
            0x2dc616d796b0ecac95c855ccb68ea147ba7ae912cc510afd22db4596056d6886,
            0x1afc010c12a7d030252d49bbe091ac357b40f62e2c1ac4c1f346b20ce3c2c5c6
        ]
    ];

    

    /**
     * @notice  Verifies a single point evaluation proof. Function name follows `ark-poly`.
     * @dev     To avoid ops in G_2, we slightly tweak how the verification is done.
     * @param   c  G_1 point commitment to polynomial.
     * @param   pi G_1 point proof.
     * @param   x  Value to prove evaluation of polynomial at.
     * @param   y  Evaluation poly(x).
     * @return  result Indicates if KZG proof is correct.
     */
    function check(uint256[2] calldata c, uint256[2] calldata pi, uint256 x, uint256 y)
        public
        view
        returns (bool result)
    {
        //
        // we want to:
        //      1. avoid gas intensive ops in G2
        //      2. format the pairing check in line with what the evm opcode expects.
        //
        // we can do this by tweaking the KZG check to be:
        //
        //          e(pi, vk - x * g2) = e(c - y * g1, g2) [initial check]
        //          e(pi, vk - x * g2) * e(c - y * g1, g2)^{-1} = 1
        //          e(pi, vk - x * g2) * e(-c + y * g1, g2) = 1 [bilinearity of pairing for all subsequent steps]
        //          e(pi, vk) * e(pi, -x * g2) * e(-c + y * g1, g2) = 1
        //          e(pi, vk) * e(-x * pi, g2) * e(-c + y * g1, g2) = 1
        //          e(pi, vk) * e(x * -pi - c + y * g1, g2) = 1 [done]
        //                        |_   rhs_pairing  _|
        //
        uint256[2] memory rhs_pairing =
            add(mulScalar(negate(pi), x), add(negate(c), mulScalar(G_1, y)));
        return pairing(pi, VK, rhs_pairing, G_2);
    }

    function evalPolyAt(uint256[] memory _coefficients, uint256 _index) public pure returns (uint256) {
        uint256 m = BN254_SCALAR_FIELD;
        uint256 result = 0;
        uint256 powerOfX = 1;

        for (uint256 i = 0; i < _coefficients.length; i++) {
            uint256 coeff = _coefficients[i];
            assembly {
                result := addmod(result, mulmod(powerOfX, coeff, m), m)
                powerOfX := mulmod(powerOfX, _index, m)
            }
        }
        return result;
    }

    
}

/* =============================== */
/* Groth16 verifier methods */
/*
    Copyright 2021 0KIMS association.

    * `solidity-verifiers` added comment
        This file is a template built out of [snarkJS](https://github.com/iden3/snarkjs) groth16 verifier.
        See the original ejs template [here](https://github.com/iden3/snarkjs/blob/master/templates/verifier_groth16.sol.ejs)
    *

    snarkJS is a free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    snarkJS is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
    or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
    License for more details.

    You should have received a copy of the GNU General Public License
    along with snarkJS. If not, see <https://www.gnu.org/licenses/>.
*/

contract Groth16Verifier {
    // Scalar field size
    uint256 constant r    = 21888242871839275222246405745257275088548364400416034343698204186575808495617;
    // Base field size
    uint256 constant q   = 21888242871839275222246405745257275088696311157297823662689037894645226208583;

    // Verification Key data
    uint256 constant alphax  = 0x0159d1c979c1e66d7dfac3ca29121ced5779b5ac197b0da615d3dfe192281280;
    uint256 constant alphay  = 0x2b30b2d3190b1a4852a06d1cf8b8d797f60177217b77ea267e04dbbe6838d3ef;
    uint256 constant betax1  = 0x27abd69011b16141682e22b3945e5bc46c509511649cc08ec374be23cbd2fafb;
    uint256 constant betax2  = 0x1e582c437b215b5450038a7e6e34f51c1af6ea4792126af059d794f83740842a;
    uint256 constant betay1  = 0x0afb1bbfcae227064b015100df0555166b6a3f450b2930c85fbe4849ce008042;
    uint256 constant betay2  = 0x265fbbab1b0b09bfc7a3c7b2cb4a9d7b8248e228a292731cd7c1ffa0b29ab006;
    uint256 constant gammax1 = 0x18b4781226247d525cd591199778d845b0147501d0551aacf2cda12b6403a74e;
    uint256 constant gammax2 = 0x25ab6e5f886f079a31efa5bd7e71200ae8e5a34410658fdaff29ea0ea9a1e2de;
    uint256 constant gammay1 = 0x29abd00ed62ede314d27c66398bb8f51679c8a7b634365cc6a8f83f77019ea83;
    uint256 constant gammay2 = 0x0e6b0ca4f6e0ff6cbfd6543b2aa2969b52524751851c4e698f149c5e083b2c7c;
    uint256 constant deltax1 = 0x17a27e9ef0148d1bbee6f9934aff893a616909eec37d97d4cb1ff61df0f84586;
    uint256 constant deltax2 = 0x24da8777da5f16c2ea75ebbeef4d4643ec7eb12f10750c69b85a3b42e610d38d;
    uint256 constant deltay1 = 0x05045fc1e3a2f4bc7312be69c5956e94f075cbf5e2913c4d9db1d68ffca174c0;
    uint256 constant deltay2 = 0x2c9feca34b88342d190069ce25b7e669feb7a65fa7bfe71672ae89c7bace1fa0;

    
    uint256 constant IC0x = 0x2bce85638bb85e515e14b9cc4a6586de80a3a6af217885273eccc754bb06e26c;
    uint256 constant IC0y = 0x044fda575138599755a627a949a428a0730d0ec60d4aedcf61e8e99deea496af;
    
    uint256 constant IC1x = 0x0baf0d2c69a540a05a20e6047b7179e4a8e3c958648f13de96eebd2c27d8829f;
    uint256 constant IC1y = 0x17b26bf723d85cabb6f537364a091590e31c888d963bae324c28e4e88ecb97fb;
    
    uint256 constant IC2x = 0x2d15ab99d26faca01d7377fc4fd5943a9b243227f8dc27a65c1021b30260b114;
    uint256 constant IC2y = 0x0ce5358ea641f6e751b985d9bda130a314d642e9d7593589436dcffea8c40792;
    
    uint256 constant IC3x = 0x267b05b0d39bc453bf38d09bdd8dfba844610f84b11df0f653059f67d124f2c4;
    uint256 constant IC3y = 0x01515a05766bd70449e24233bfd2800115ffcb21f47e7e30ef13d1ceacb67a50;
    
    uint256 constant IC4x = 0x0b21a7bef236d87b39ca8ca515d872ddbcec910fc240c3f87f730194aa3475cb;
    uint256 constant IC4y = 0x1d8f64be37036bc9891c1cf24c0243f97efa7e2efe1b2a99aaee86835a7d51d4;
    
    uint256 constant IC5x = 0x02979be182b4f202b2677fc8301acdae92a0f0bdacd0c0e3921ed1eeeb83b0bf;
    uint256 constant IC5y = 0x2cd9e8390c113bbaaf208edfa8533ce03da25167f5909ea698df646da70ace4a;
    
    uint256 constant IC6x = 0x0da846d9dc28e679b066f5fbab2e207e24a1af1c5066940a9c6221319633e4b9;
    uint256 constant IC6y = 0x0347aa37731528bba8cae81c14043e5234802f145d9f7a3ff1905bed59ee7e60;
    
    uint256 constant IC7x = 0x1170168392b55a66102d0e6cb559a602c6961e645a5fd654ab5ada05bd99eb1a;
    uint256 constant IC7y = 0x1fcda6cedf35ce015cfe700b69e65f62b13feadbf395700c8b7310e517ff029b;
    
    uint256 constant IC8x = 0x22334a09dd3961f6bac6ee9a232254cb88144e7bb69294ab7c4957c35a20f788;
    uint256 constant IC8y = 0x186c0cc5cbc7e451313905ce5267bc8600bf592263059c3625dd3ee429574c0b;
    
    uint256 constant IC9x = 0x1ea61334cfc7a573e0e631cf046b482ff093438fb3e51613d9ea715ddd8da038;
    uint256 constant IC9y = 0x2ab0aa5f2c73456fc16584941194ad3b4e095b1d6a6fb5b30f1ab24b942942c6;
    
    uint256 constant IC10x = 0x0e1f2b2f0d7fd07a2cbeda15b4e92f406db9b855ded9716f7177e7a09ecce7d7;
    uint256 constant IC10y = 0x0a2008c65e5adb33c1695fd560933ee21fc4cd48391f708527310ff54814134a;
    
    uint256 constant IC11x = 0x28e9d8b99f16780ea12d2fcc3c0754095802408a70f44762e025a925811128b1;
    uint256 constant IC11y = 0x0dd27b78c66ddc17ee12410c74f1ec02316b2bdca1463dd985aa28e4c18e9c2c;
    
    uint256 constant IC12x = 0x0ac8002002c8d7daa03796e41ae72425888bdd78b2c9374804ed2a5b25eabca8;
    uint256 constant IC12y = 0x135bc1c06ee7d31a33ae2aa8dde7c6e2d8c03e5ad2c5b5b5a053ebb620bf493c;
    
    uint256 constant IC13x = 0x2736d2e43c8cb1d098caa5b98df7dc819521909f0caaeed423bfbc51195ef304;
    uint256 constant IC13y = 0x071cacfbf63af006494df539eade0ed39c74a10604eb2db3331b8999b12eae9a;
    
    uint256 constant IC14x = 0x1db3afcc80fad50fa3ccac8e89b8145665c285fb73ec17751f82856a4e42166a;
    uint256 constant IC14y = 0x1f61a39aba2b10e6cff267871fab23afcba732b2ea2909f8b4b482b9d771d871;
    
    uint256 constant IC15x = 0x2b697dd6a9c2e9f0cfb104896c7805da24bac98e5f05251a1963d0ebeb542e65;
    uint256 constant IC15y = 0x13b7b2a7712c76a2c8d7cadc5d684faf5b98b39c41a05d1dece0861fa9650be5;
    
    uint256 constant IC16x = 0x1fe24414937a2e87abd0a653e7d4363064eaab1506357907d216ffc0125f96e2;
    uint256 constant IC16y = 0x23d73c7b9bb3d36d61192414ce9b26379618db80efdb5ad7f95378d95bd4c5ee;
    
    uint256 constant IC17x = 0x0269172bcc412751fafe8918fff12d5e2ba5612e5eb1dac53c83905b826b1685;
    uint256 constant IC17y = 0x270e63dc18960bf7733df04b91b704e7c9e7398c3deac964a145898add634a66;
    
    uint256 constant IC18x = 0x130ddd6ab658f97efdb30d38f223e8c485c603070b61ad5aa489f958e8f8fd90;
    uint256 constant IC18y = 0x0fe80317204389460e00efceeeeea8aeda602a4d422175cc497a52ef5d387723;
    
    uint256 constant IC19x = 0x146cfac2a141383a31a79ec2e9851f9e3058286e8e55703a87ccf24fe9e8e437;
    uint256 constant IC19y = 0x2c1b578e8c58e61072549d1d0662b16e095bb2dabfe374c65435482a8c7b254c;
    
    uint256 constant IC20x = 0x0b8fa9f1d24963752f227d6aa77481c0b1435d3836fc848f433f368d45abfd03;
    uint256 constant IC20y = 0x10d0aa2aced06fecfcb007ec37401c3c9b49b33267ad9e0d46c18ae88f288905;
    
    uint256 constant IC21x = 0x2b4ab4185aa6db822d3a66ce329a46e0df9c29dee8967b468c76acaa86628e4c;
    uint256 constant IC21y = 0x281a23957e01380759a9dd7668ead63f6c9fab86094d14a3851a2f485ea50fb2;
    
    uint256 constant IC22x = 0x05a88f77545e460b76158ab9044c9b2470ac7d55f6275f68e1c784c91dbd61c5;
    uint256 constant IC22y = 0x1b6560b47dbf821564534469b352f90aaa6bbe691f57332a1a106637fca27998;
    
    uint256 constant IC23x = 0x272c491dd26ef6676044363306b033aaf576fc41cdabf86256f4e80faf6321a6;
    uint256 constant IC23y = 0x16570e50babe551e62e684247ba1e1d1fa673ec3cad0f0448a256f1a622f9616;
    
    uint256 constant IC24x = 0x21bf95a10eec4e30f1dec8c9c2ce6df64447e3c70069b3e011a135313f4d7a8d;
    uint256 constant IC24y = 0x20dc612cbeb4a4d7fbf656c2a984ce462f7c86f8fc832af0bfdfe420971b6354;
    
    uint256 constant IC25x = 0x0a0379bcf4e83a7474265361bd122058a9841ace8fdccb59ffd797d11ee5ba82;
    uint256 constant IC25y = 0x04d3baab75e398c4fb3c7b8921320bedf8c312c378ab5225cb98bafa9fdeccfe;
    
    uint256 constant IC26x = 0x2b7ea1e78a0ece8d7ebe3634d0c7b497d7b96e7e50bd202c0d7edc552f117ca5;
    uint256 constant IC26y = 0x13c5bc814886a9cb406a53d45e1d3f31b68b206e160a422d2447426a7c716b5f;
    
    uint256 constant IC27x = 0x1ff967948889c1adebfdd1519cb0b258d57f6ce2816992fbd645526df2d2a1db;
    uint256 constant IC27y = 0x106877efb8026b3e45973896b26ca37c8bd3b7ab539d43e97f8195f1ec23b237;
    
    uint256 constant IC28x = 0x08d8d1e6695fb4e47ec04aec2303324baa51200f4a8e84c22d07da1622403b9d;
    uint256 constant IC28y = 0x065bb5266ee317eda88494c064b5b05d27a2d15e5de296bea3e5b40048e48e0c;
    
    uint256 constant IC29x = 0x1134842ff7dde2cc84a964ce5205c8dfa5008920f64ef37046244cd065787876;
    uint256 constant IC29y = 0x26adf68698a43749d581fb6dd16110cca9a0067a786950089efaf6f32bb6a395;
    
    uint256 constant IC30x = 0x097e8a29142fb3a4dc590ed068bbad3b3bb7182e19c07d16f21db3bd781b3860;
    uint256 constant IC30y = 0x1bd4be10958ab522f59e11950cf68ab2aeaf1b14ef5fbf0f01eedbfb1f896b5c;
    
    uint256 constant IC31x = 0x01f935e27a95b967c1b7456ec740772aace605e14b7d3823c9d7a181d2d846be;
    uint256 constant IC31y = 0x2cf224523b53c3e6d769374814000707a2e4027de76ce90f00dc78ff519f52ff;
    
    uint256 constant IC32x = 0x2df6abac85bb60385ee8e0c12362d46a5721d49c165679493eb026cb25f83a7f;
    uint256 constant IC32y = 0x211b8eda8b52870b7b7ef86499489a91323f03617cea37f8eb3b74c077ce1d78;
    
    
    // Memory data
    uint16 constant pVk = 0;
    uint16 constant pPairing = 128;

    uint16 constant pLastMem = 896;

    function verifyProof(uint[2] calldata _pA, uint[2][2] calldata _pB, uint[2] calldata _pC, uint[32] calldata _pubSignals) public view returns (bool) {
        assembly {
            function checkField(v) {
                if iszero(lt(v, r)) {
                    mstore(0, 0)
                    return(0, 0x20)
                }
            }
            
            // G1 function to multiply a G1 value(x,y) to value in an address
            function g1_mulAccC(pR, x, y, s) {
                let success
                let mIn := mload(0x40)
                mstore(mIn, x)
                mstore(add(mIn, 32), y)
                mstore(add(mIn, 64), s)

                success := staticcall(sub(gas(), 2000), 7, mIn, 96, mIn, 64)

                if iszero(success) {
                    mstore(0, 0)
                    return(0, 0x20)
                }

                mstore(add(mIn, 64), mload(pR))
                mstore(add(mIn, 96), mload(add(pR, 32)))

                success := staticcall(sub(gas(), 2000), 6, mIn, 128, pR, 64)

                if iszero(success) {
                    mstore(0, 0)
                    return(0, 0x20)
                }
            }

            function checkPairing(pA, pB, pC, pubSignals, pMem) -> isOk {
                let _pPairing := add(pMem, pPairing)
                let _pVk := add(pMem, pVk)

                mstore(_pVk, IC0x)
                mstore(add(_pVk, 32), IC0y)

                // Compute the linear combination vk_x
                
                
                g1_mulAccC(_pVk, IC1x, IC1y, calldataload(add(pubSignals, 0)))
                g1_mulAccC(_pVk, IC2x, IC2y, calldataload(add(pubSignals, 32)))
                g1_mulAccC(_pVk, IC3x, IC3y, calldataload(add(pubSignals, 64)))
                g1_mulAccC(_pVk, IC4x, IC4y, calldataload(add(pubSignals, 96)))
                g1_mulAccC(_pVk, IC5x, IC5y, calldataload(add(pubSignals, 128)))
                g1_mulAccC(_pVk, IC6x, IC6y, calldataload(add(pubSignals, 160)))
                g1_mulAccC(_pVk, IC7x, IC7y, calldataload(add(pubSignals, 192)))
                g1_mulAccC(_pVk, IC8x, IC8y, calldataload(add(pubSignals, 224)))
                g1_mulAccC(_pVk, IC9x, IC9y, calldataload(add(pubSignals, 256)))
                g1_mulAccC(_pVk, IC10x, IC10y, calldataload(add(pubSignals, 288)))
                g1_mulAccC(_pVk, IC11x, IC11y, calldataload(add(pubSignals, 320)))
                g1_mulAccC(_pVk, IC12x, IC12y, calldataload(add(pubSignals, 352)))
                g1_mulAccC(_pVk, IC13x, IC13y, calldataload(add(pubSignals, 384)))
                g1_mulAccC(_pVk, IC14x, IC14y, calldataload(add(pubSignals, 416)))
                g1_mulAccC(_pVk, IC15x, IC15y, calldataload(add(pubSignals, 448)))
                g1_mulAccC(_pVk, IC16x, IC16y, calldataload(add(pubSignals, 480)))
                g1_mulAccC(_pVk, IC17x, IC17y, calldataload(add(pubSignals, 512)))
                g1_mulAccC(_pVk, IC18x, IC18y, calldataload(add(pubSignals, 544)))
                g1_mulAccC(_pVk, IC19x, IC19y, calldataload(add(pubSignals, 576)))
                g1_mulAccC(_pVk, IC20x, IC20y, calldataload(add(pubSignals, 608)))
                g1_mulAccC(_pVk, IC21x, IC21y, calldataload(add(pubSignals, 640)))
                g1_mulAccC(_pVk, IC22x, IC22y, calldataload(add(pubSignals, 672)))
                g1_mulAccC(_pVk, IC23x, IC23y, calldataload(add(pubSignals, 704)))
                g1_mulAccC(_pVk, IC24x, IC24y, calldataload(add(pubSignals, 736)))
                g1_mulAccC(_pVk, IC25x, IC25y, calldataload(add(pubSignals, 768)))
                g1_mulAccC(_pVk, IC26x, IC26y, calldataload(add(pubSignals, 800)))
                g1_mulAccC(_pVk, IC27x, IC27y, calldataload(add(pubSignals, 832)))
                g1_mulAccC(_pVk, IC28x, IC28y, calldataload(add(pubSignals, 864)))
                g1_mulAccC(_pVk, IC29x, IC29y, calldataload(add(pubSignals, 896)))
                g1_mulAccC(_pVk, IC30x, IC30y, calldataload(add(pubSignals, 928)))
                g1_mulAccC(_pVk, IC31x, IC31y, calldataload(add(pubSignals, 960)))
                g1_mulAccC(_pVk, IC32x, IC32y, calldataload(add(pubSignals, 992)))

                // -A
                mstore(_pPairing, calldataload(pA))
                mstore(add(_pPairing, 32), mod(sub(q, calldataload(add(pA, 32))), q))

                // B
                mstore(add(_pPairing, 64), calldataload(pB))
                mstore(add(_pPairing, 96), calldataload(add(pB, 32)))
                mstore(add(_pPairing, 128), calldataload(add(pB, 64)))
                mstore(add(_pPairing, 160), calldataload(add(pB, 96)))

                // alpha1
                mstore(add(_pPairing, 192), alphax)
                mstore(add(_pPairing, 224), alphay)

                // beta2
                mstore(add(_pPairing, 256), betax1)
                mstore(add(_pPairing, 288), betax2)
                mstore(add(_pPairing, 320), betay1)
                mstore(add(_pPairing, 352), betay2)

                // vk_x
                mstore(add(_pPairing, 384), mload(add(pMem, pVk)))
                mstore(add(_pPairing, 416), mload(add(pMem, add(pVk, 32))))


                // gamma2
                mstore(add(_pPairing, 448), gammax1)
                mstore(add(_pPairing, 480), gammax2)
                mstore(add(_pPairing, 512), gammay1)
                mstore(add(_pPairing, 544), gammay2)

                // C
                mstore(add(_pPairing, 576), calldataload(pC))
                mstore(add(_pPairing, 608), calldataload(add(pC, 32)))

                // delta2
                mstore(add(_pPairing, 640), deltax1)
                mstore(add(_pPairing, 672), deltax2)
                mstore(add(_pPairing, 704), deltay1)
                mstore(add(_pPairing, 736), deltay2)


                let success := staticcall(sub(gas(), 2000), 8, _pPairing, 768, _pPairing, 0x20)


                isOk := and(success, mload(_pPairing))
            }

            let pMem := mload(0x40)
            mstore(0x40, add(pMem, pLastMem))

            // Validate that all evaluations ∈ F
            
            checkField(calldataload(add(_pubSignals, 0)))
            
            checkField(calldataload(add(_pubSignals, 32)))
            
            checkField(calldataload(add(_pubSignals, 64)))
            
            checkField(calldataload(add(_pubSignals, 96)))
            
            checkField(calldataload(add(_pubSignals, 128)))
            
            checkField(calldataload(add(_pubSignals, 160)))
            
            checkField(calldataload(add(_pubSignals, 192)))
            
            checkField(calldataload(add(_pubSignals, 224)))
            
            checkField(calldataload(add(_pubSignals, 256)))
            
            checkField(calldataload(add(_pubSignals, 288)))
            
            checkField(calldataload(add(_pubSignals, 320)))
            
            checkField(calldataload(add(_pubSignals, 352)))
            
            checkField(calldataload(add(_pubSignals, 384)))
            
            checkField(calldataload(add(_pubSignals, 416)))
            
            checkField(calldataload(add(_pubSignals, 448)))
            
            checkField(calldataload(add(_pubSignals, 480)))
            
            checkField(calldataload(add(_pubSignals, 512)))
            
            checkField(calldataload(add(_pubSignals, 544)))
            
            checkField(calldataload(add(_pubSignals, 576)))
            
            checkField(calldataload(add(_pubSignals, 608)))
            
            checkField(calldataload(add(_pubSignals, 640)))
            
            checkField(calldataload(add(_pubSignals, 672)))
            
            checkField(calldataload(add(_pubSignals, 704)))
            
            checkField(calldataload(add(_pubSignals, 736)))
            
            checkField(calldataload(add(_pubSignals, 768)))
            
            checkField(calldataload(add(_pubSignals, 800)))
            
            checkField(calldataload(add(_pubSignals, 832)))
            
            checkField(calldataload(add(_pubSignals, 864)))
            
            checkField(calldataload(add(_pubSignals, 896)))
            
            checkField(calldataload(add(_pubSignals, 928)))
            
            checkField(calldataload(add(_pubSignals, 960)))
            
            checkField(calldataload(add(_pubSignals, 992)))
            
            checkField(calldataload(add(_pubSignals, 1024)))
            

            // Validate all evaluations
            let isValid := checkPairing(_pA, _pB, _pC, _pubSignals, pMem)

            mstore(0, isValid)
            
            return(0, 0x20)
        }
    }
}


/* =============================== */
/* Nova+CycleFold Decider verifier */
/**
 * @notice  Computes the decomposition of a `uint256` into num_limbs limbs of bits_per_limb bits each.
 * @dev     Compatible with sonobe::folding-schemes::folding::circuits::nonnative::nonnative_field_to_field_elements.
 */
library LimbsDecomposition {
    function decompose(uint256 x) internal pure returns (uint256[4] memory) {
        uint256[4] memory limbs;
        for (uint8 i = 0; i < 4; i++) {
            limbs[i] = (x >> (64 * i)) & ((1 << 64) - 1);
        }
        return limbs;
    }
}

/**
 * @author  PSE & 0xPARC
 * @title   NovaDecider contract, for verifying Nova IVC SNARK proofs.
 * @dev     This is an askama template which, when templated, features a Groth16 and KZG10 verifiers from which this contract inherits.
 */
contract NovaDecider is Groth16Verifier, KZG10Verifier {
    /**
     * @notice  Computes the linear combination of a and b with r as the coefficient.
     * @dev     All ops are done mod the BN254 scalar field prime
     */
    function rlc(uint256 a, uint256 r, uint256 b) internal pure returns (uint256 result) {
        assembly {
            result := addmod(a, mulmod(r, b, BN254_SCALAR_FIELD), BN254_SCALAR_FIELD)
        }
    }

    /**
     * @notice  Verifies a nova cyclefold proof consisting of two KZG proofs and of a groth16 proof.
     * @dev     The selector of this function is "dynamic", since it depends on `z_len`.
     */
    function verifyNovaProof(
        // inputs are grouped to prevent errors due stack too deep
        uint256[3] calldata i_z0_zi, // [i, z0, zi] where |z0| == |zi|
        uint256[4] calldata U_i_cmW_U_i_cmE, // [U_i_cmW[2], U_i_cmE[2]]
        uint256[2] calldata u_i_cmW, // [u_i_cmW[2]]
        uint256[3] calldata cmT_r, // [cmT[2], r]
        uint256[2] calldata pA, // groth16 
        uint256[2][2] calldata pB, // groth16
        uint256[2] calldata pC, // groth16
        uint256[4] calldata challenge_W_challenge_E_kzg_evals, // [challenge_W, challenge_E, eval_W, eval_E]
        uint256[2][2] calldata kzg_proof // [proof_W, proof_E]
    ) public view returns (bool) {

        require(i_z0_zi[0] >= 2, "Folding: the number of folded steps should be at least 2");

        // from gamma_abc_len, we subtract 1. 
        uint256[32] memory public_inputs; 

        public_inputs[0] = 0x01a6f22c6a14fb042ca035c458c31fa54d7606dd1c04c8ca1cfcdaef6930926b;
        public_inputs[1] = i_z0_zi[0];

        for (uint i = 0; i < 2; i++) {
            public_inputs[2 + i] = i_z0_zi[1 + i];
        }

        {
            // U_i.cmW + r * u_i.cmW
            uint256[2] memory mulScalarPoint = super.mulScalar([u_i_cmW[0], u_i_cmW[1]], cmT_r[2]);
            uint256[2] memory cmW = super.add([U_i_cmW_U_i_cmE[0], U_i_cmW_U_i_cmE[1]], mulScalarPoint);

            {
                uint256[4] memory cmW_x_limbs = LimbsDecomposition.decompose(cmW[0]);
                uint256[4] memory cmW_y_limbs = LimbsDecomposition.decompose(cmW[1]);
        
                for (uint8 k = 0; k < 4; k++) {
                    public_inputs[4 + k] = cmW_x_limbs[k];
                    public_inputs[8 + k] = cmW_y_limbs[k];
                }
            }
        
            require(this.check(cmW, kzg_proof[0], challenge_W_challenge_E_kzg_evals[0], challenge_W_challenge_E_kzg_evals[2]), "KZG: verifying proof for challenge W failed");
        }

        {
            // U_i.cmE + r * cmT
            uint256[2] memory mulScalarPoint = super.mulScalar([cmT_r[0], cmT_r[1]], cmT_r[2]);
            uint256[2] memory cmE = super.add([U_i_cmW_U_i_cmE[2], U_i_cmW_U_i_cmE[3]], mulScalarPoint);

            {
                uint256[4] memory cmE_x_limbs = LimbsDecomposition.decompose(cmE[0]);
                uint256[4] memory cmE_y_limbs = LimbsDecomposition.decompose(cmE[1]);
            
                for (uint8 k = 0; k < 4; k++) {
                    public_inputs[12 + k] = cmE_x_limbs[k];
                    public_inputs[16 + k] = cmE_y_limbs[k];
                }
            }

            require(this.check(cmE, kzg_proof[1], challenge_W_challenge_E_kzg_evals[1], challenge_W_challenge_E_kzg_evals[3]), "KZG: verifying proof for challenge E failed");
        }

        {
            // add challenges
            public_inputs[20] = challenge_W_challenge_E_kzg_evals[0];
            public_inputs[21] = challenge_W_challenge_E_kzg_evals[1];
            public_inputs[22] = challenge_W_challenge_E_kzg_evals[2];
            public_inputs[23] = challenge_W_challenge_E_kzg_evals[3];

            uint256[4] memory cmT_x_limbs;
            uint256[4] memory cmT_y_limbs;
        
            cmT_x_limbs = LimbsDecomposition.decompose(cmT_r[0]);
            cmT_y_limbs = LimbsDecomposition.decompose(cmT_r[1]);
        
            for (uint8 k = 0; k < 4; k++) {
                public_inputs[20 + 4 + k] = cmT_x_limbs[k]; 
                public_inputs[24 + 4 + k] = cmT_y_limbs[k];
            }

            bool success_g16 = this.verifyProof(pA, pB, pC, public_inputs);
            require(success_g16 == true, "Groth16: verifying proof failed");
        }

        return(true);
    }
}