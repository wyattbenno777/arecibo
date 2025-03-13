alias BaseField = BigInt256;
alias ScalarField = BigInt256;

const BASE_MODULUS: BigInt256 = BigInt256(array(65535u, 1u, 17377u, 58629u, 31690u, 60064u, 17038u, 40814u, 20717u, 4814u, 10775u, 48118u, 35964u, 29867u, 56439u, 2u));

const BASE_MODULUS_MEDIUM_WIDE: BigInt272 = BigInt272(array(65535u, 1u, 17377u, 58629u, 31690u, 60064u, 17038u, 40814u, 20717u, 4814u, 10775u, 48118u, 35964u, 29867u, 56439u, 2u, 0u));

const BASE_MODULUS_WIDE: BigInt512 = BigInt512(array(65535u, 1u, 17377u, 58629u, 31690u, 60064u, 17038u, 40814u, 20717u, 4814u, 10775u, 48118u, 35964u, 29867u, 56439u, 2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u));

const BASE_NBITS = 255;

const BASE_M: BigInt256 = BigInt256(array(4558u, 61268u, 6391u, 51638u, 1959u, 3734u, 16655u, 62282u, 33496u, 22191u, 62269u, 56316u, 63923u, 30752u, 62416u, 15u));

const ZERO: BigInt256 = BigInt256(array(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u));

const ONE: BigInt256 = BigInt256(array(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u));

fn get_higher_with_slack(a: BigInt512) -> BaseField {
    var out: BaseField;
    const slack = L - BASE_NBITS;
    for (var i = 0u; i < N; i = i + 1u) {
        out.limbs[i] = ((a.limbs[i + N] << slack) + (a.limbs[i + N - 1] >> (W - slack))) & W_mask;
    }
    return out;
}

// once reduces once (assumes that 0 <= a < 2 * mod)
fn field_reduce(a: BigInt256) -> BaseField {
    var res: BigInt256;
    var underflow = sub(a, BASE_MODULUS, & res);
    if (underflow == 1u) {
        return a;
    }
    else {
        return res;
    }
}

fn shorten(a: BigInt272) -> BigInt256 {
    var out: BigInt256;
    for (var i = 0u; i < N; i = i + 1u) {
        out.limbs[i] = a.limbs[i];
    }
    return out;
}

// reduces l times (assumes that 0 <= a < multi * mod)
fn field_reduce_272(a: BigInt272, multi: u32) -> BaseField {
    var res: BigInt272;
    var cur = a;
    var cur_multi = multi + 1;
    while (cur_multi > 0u) {
        var underflow = sub_272(cur, BASE_MODULUS_MEDIUM_WIDE, & res);
        if (underflow == 1u) {
            return shorten(cur);
        }
        else {
            cur = res;
        }
        cur_multi = cur_multi - 1u;
    }
    return ZERO;
}

fn field_add(a: BaseField, b: BaseField) -> BaseField {
    var res: BaseField;
    add(a, b, & res);
    return field_reduce(res);
}

fn field_sub(a: BaseField, b: BaseField) -> BaseField {
    var res: BaseField;
    var carry = sub(a, b, & res);
    if (carry == 0u) {
        return res;
    }
    add(res, BASE_MODULUS, & res);
    return res;
}

fn field_mul(a: BaseField, b: BaseField) -> BaseField {
    var xy: BigInt512 = mul(a, b);
    var xy_hi: BaseField = get_higher_with_slack(xy);
    var l: BigInt512 = mul(xy_hi, BASE_M);
    var l_hi: BaseField = get_higher_with_slack(l);
    var lp: BigInt512 = mul(l_hi, BASE_MODULUS);
    var r_wide: BigInt512;
    sub_512(xy, lp, & r_wide);

    var r_wide_reduced: BigInt512;
    var underflow = sub_512(r_wide, BASE_MODULUS_WIDE, & r_wide_reduced);
    if (underflow == 0u) {
        r_wide = r_wide_reduced;
    }
    var r: BaseField;
    for (var i = 0u; i < N; i = i + 1u) {
        r.limbs[i] = r_wide.limbs[i];
    }
    return field_reduce(r);
}

fn field_small_scalar_shift(l: u32, a: BaseField) -> BaseField {
    // max shift allowed is 16
    // assert (l < 16u);
    var res: BigInt272;
    for (var i = 0u; i < N; i = i + 1u) {
        let shift = a.limbs[i] << l;
        res.limbs[i] = res.limbs[i] | (shift & W_mask);
        res.limbs[i + 1] = (shift >> W);
    }

    var output = field_reduce_272(res, (1u << l));
    // can probably be optimised
    return output;
}

fn field_pow(p: BaseField, e: u32) -> BaseField {
    var res: BaseField = p;
    for (var i = 1u; i < e; i = i + 1u) {
        res = field_mul(res, p);
    }
    return res;
}

fn field_eq(a: BaseField, b: BaseField) -> bool {
    for (var i = 0u; i < N; i = i + 1u) {
        if (a.limbs[i] != b.limbs[i]) {
            return false;
        }
    }
    return true;
}

fn field_sqr(a: BaseField) -> BaseField {
    var xy: BigInt512 = sqr(a);
    var xy_hi: BaseField = get_higher_with_slack(xy);
    var l: BigInt512 = mul(xy_hi, BASE_M);
    var l_hi: BaseField = get_higher_with_slack(l);
    var lp: BigInt512 = mul(l_hi, BASE_MODULUS);
    var r_wide: BigInt512;
    sub_512(xy, lp, & r_wide);

    var r_wide_reduced: BigInt512;
    var underflow = sub_512(r_wide, BASE_MODULUS_WIDE, & r_wide_reduced);
    if (underflow == 0u) {
        r_wide = r_wide_reduced;
    }
    var r: BaseField;
    for (var i = 0u; i < N; i = i + 1u) {
        r.limbs[i] = r_wide.limbs[i];
    }
    return field_reduce(r);
}


