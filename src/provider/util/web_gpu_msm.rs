use ff::PrimeField;
use halo2curves::CurveAffine;
use js_sys::{BigInt, Object, Promise, Reflect};
use num_bigint::BigUint;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::console;

#[wasm_bindgen(module = "/src/provider/util/bestMSM.js")]
extern "C" {
  #[wasm_bindgen(catch)]
  fn webgpu_best_msm(bases: Vec<JsValue>, scalars: Vec<JsValue>) -> Result<Promise, JsValue>;
}


#[wasm_bindgen]
#[derive(Debug)]
pub struct MsmResult {
  x: JsValue,
  y: JsValue,
}

#[wasm_bindgen]
impl MsmResult {
  #[wasm_bindgen(constructor)]
  pub fn new(x: JsValue, y: JsValue) -> MsmResult {
    MsmResult { x, y }
  }
  #[wasm_bindgen(getter)]
  pub fn x(&self) -> JsValue {
    self.x.clone()
  }
  #[wasm_bindgen(getter)]
  pub fn y(&self) -> JsValue {
    self.y.clone()
  }
}

#[wasm_bindgen]
pub async fn run_gpu_msm(bases: Vec<JsValue>, scalars: Vec<JsValue>) -> Result<MsmResult, JsValue> {
  let promise = webgpu_best_msm(bases, scalars);
  match promise {
    Ok(promise) => {
      let js_value = JsFuture::from(promise).await?;
      let x = Reflect::get(&js_value, &JsValue::from_str("x"))?;
      let y = Reflect::get(&js_value, &JsValue::from_str("y"))?;
      let result = MsmResult::new(x, y);
      Ok(result)
    }
    Err(e) => {
      console::log_1(&format!("Error: {:?}", e).into());
      Err(e)
    }
  }
}

pub fn jsvalue_to_primefield<F: PrimeField>(value: JsValue) -> Result<F, JsValue> {
  let bigint = BigInt::from(value);

  let bigint_str = bigint.to_string(10)?;

  let str = bigint_str.as_string().unwrap();

  let f = F::from_str_vartime(&str).unwrap();
  Ok(f)
}

pub fn curve_to_js_point<C: CurveAffine>(point: C) -> JsValue {
  let obj = Object::new();
  let coordinates = point.coordinates().unwrap();
  let x = coordinates.x();
  let y = coordinates.y();
  // The inputs are affine points, so the z property will be 1n. 
  // The t property is the field multiplication of x by y.
  let t = *x * y;
  Reflect::set(&obj, &"x".into(), &primefield_to_bigint_js(*x)).unwrap();
  Reflect::set(&obj, &"y".into(), &primefield_to_bigint_js(*y)).unwrap();
  Reflect::set(&obj, &"t".into(), &primefield_to_bigint_js(t)).unwrap();
  Reflect::set(&obj, &"z".into(), &JsValue::bigint_from_str("1")).unwrap();
  obj.into()
}

pub fn primefield_to_bigint_js<F: PrimeField>(value: F) -> JsValue {
  let s_bytes = value.to_repr();
  let s_bytes_ref = s_bytes.as_ref(); 
  let s_biguint = BigUint::from_bytes_le(s_bytes_ref);
  let s_str = s_biguint.to_str_radix(10);
  JsValue::bigint_from_str(&s_str)
}

// #[cfg(target_arch = "wasm32")]
#[cfg(test)]
mod tests {
  use super::*;
  use halo2curves::bn256::Fr;
  use rand::thread_rng;
  use rand::Rng;
  use wasm_bindgen_test::wasm_bindgen_test;
  
  #[wasm_bindgen_test]
  fn test_primefield_to_bigint_js_roundtrip() {
    let mut rng = thread_rng();
    let random_u64: u64 = rng.gen();
    let f = Fr::from(random_u64);
    let bigint = primefield_to_bigint_js(f);
    let f_roundtrip = jsvalue_to_primefield::<Fr>(bigint).unwrap();
    assert_eq!(f, f_roundtrip);
  }
}
