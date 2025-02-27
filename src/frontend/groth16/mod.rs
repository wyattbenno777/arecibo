//! The [Groth16] proving system.
//!
//! [Groth16]: https://eprint.iacr.org/2016/260

pub mod aggregate;
mod ext;
mod generator;
#[cfg(not(target_arch = "wasm32"))]
mod mapped_params;
mod multiexp;
mod params;
mod proof;
mod prover;
mod verifier;
mod verifying_key;

mod multiscalar;

#[cfg(not(target_arch = "wasm32"))]
pub use self::mapped_params::*;
pub use self::{ext::*, generator::*, params::*, proof::*, verifier::*, verifying_key::*};

#[cfg(test)]
mod tests {
  use crate::frontend::{
    groth16::{
      create_random_proof, generate_random_parameters, prepare_verifying_key, verify_proof,
    }, num::AllocatedNum, Circuit, ConstraintSystem, SynthesisError
  };
  use halo2curves::bn256::{Bn256, Fr};

  struct TrivialCircuit {
    a: Option<Fr>,
    b: Option<Fr>,
    c: Option<Fr>,
  }

  impl Circuit<Fr> for TrivialCircuit {
    fn synthesize<CS: ConstraintSystem<Fr>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
      // Allocate the variables for a, b, and c
      let a = cs.alloc(|| "a", || self.a.ok_or(SynthesisError::AssignmentMissing))?;
      let b = cs.alloc(|| "b", || self.b.ok_or(SynthesisError::AssignmentMissing))?;
      let c = AllocatedNum::alloc(cs.namespace(|| "c"), || Ok(self.c.unwrap()))?;
      c.inputize(cs.namespace(|| "c"))?;
      // Enforce the constraint a * b = c
      cs.enforce(|| "a * b = c", |lc| lc + a, |lc| lc + b, |lc| lc + c.get_variable());

      Ok(())
    }
  }

  #[test]
  fn test_groth16() {
    // Example values for a, b, and c
    let a = Fr::from(2);
    let b = Fr::from(3);
    let c = Fr::from(6); // because 2 * 3 = 6

    // Create an instance of the circuit
    let circuit = TrivialCircuit {
      a: Some(a),
      b: Some(b),
      c: Some(c),
    };

    // Generate the parameters for the proving system
    let rng = &mut rand::thread_rng();
    let params = generate_random_parameters::<Bn256, _, _>(circuit, rng).unwrap();
    let circuit = TrivialCircuit {
      a: Some(a),
      b: Some(b),
      c: Some(c),
    };
    // Create a proof
    let proof = create_random_proof(circuit, &params, rng).unwrap();

    // Prepare the verifying key
    let pvk = prepare_verifying_key(&params.vk);

    // Verify the proof
    let verified = verify_proof(&pvk, &proof, &[c]).unwrap();

    assert!(verified);
  }
}
