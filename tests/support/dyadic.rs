//! Exact dyadic rationals (`±mantissa * 2^exp`) for checking interval enclosures.
//!
//! Every finite f64 is dyadic and neg/add/sub/mul keep results dyadic, so the real-number
//! value of an interval kernel at a point is computed without rounding. Magnitudes are
//! little-endian `u64` limbs with no trailing zero limb (zero is the empty vector).

use std::cmp::Ordering;

type Mag = Vec<u64>;

fn trim(mut m: Mag) -> Mag {
    while m.last() == Some(&0) {
        m.pop();
    }
    m
}

fn cmp_mag(a: &Mag, b: &Mag) -> Ordering {
    a.len().cmp(&b.len()).then_with(|| {
        a.iter()
            .rev()
            .zip(b.iter().rev())
            .map(|(x, y)| x.cmp(y))
            .find(|o| *o != Ordering::Equal)
            .unwrap_or(Ordering::Equal)
    })
}

fn add_mag(a: &Mag, b: &Mag) -> Mag {
    let (a, b) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut out = Vec::with_capacity(a.len() + 1);
    let mut carry = false;
    for (i, &x) in a.iter().enumerate() {
        let (s, c1) = x.overflowing_add(b.get(i).copied().unwrap_or(0));
        let (s, c2) = s.overflowing_add(carry as u64);
        out.push(s);
        carry = c1 || c2;
    }
    if carry {
        out.push(1);
    }
    out
}

/// `a - b`, requires `a >= b`.
fn sub_mag(a: &Mag, b: &Mag) -> Mag {
    let mut out = Vec::with_capacity(a.len());
    let mut borrow = false;
    for (i, &x) in a.iter().enumerate() {
        let (d, b1) = x.overflowing_sub(b.get(i).copied().unwrap_or(0));
        let (d, b2) = d.overflowing_sub(borrow as u64);
        out.push(d);
        borrow = b1 || b2;
    }
    assert!(!borrow, "sub_mag underflow");
    trim(out)
}

fn mul_mag(a: &Mag, b: &Mag) -> Mag {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut out = vec![0u64; a.len() + b.len()];
    for (i, &x) in a.iter().enumerate() {
        let mut carry = 0u128;
        for (j, &y) in b.iter().enumerate() {
            let t = out[i + j] as u128 + x as u128 * y as u128 + carry;
            out[i + j] = t as u64;
            carry = t >> 64;
        }
        out[i + b.len()] = carry as u64;
    }
    trim(out)
}

fn shl_mag(a: &Mag, bits: u64) -> Mag {
    if a.is_empty() {
        return vec![];
    }
    let (limbs, bits) = ((bits / 64) as usize, (bits % 64) as u32);
    let mut out = vec![0u64; limbs];
    let mut carry = 0u64;
    for &x in a {
        out.push(if bits == 0 { x } else { (x << bits) | carry });
        carry = if bits == 0 { 0 } else { x >> (64 - bits) };
    }
    if carry != 0 {
        out.push(carry);
    }
    out
}

/// An exact dyadic rational. Zero has an empty magnitude and a meaningless exponent.
#[derive(Clone, Debug)]
pub struct Dyadic {
    negative: bool,
    mag: Mag,
    exp: i64,
}

impl Dyadic {
    pub fn zero() -> Self {
        Dyadic {
            negative: false,
            mag: vec![],
            exp: 0,
        }
    }

    pub fn from_i64(x: i64) -> Self {
        Dyadic {
            negative: x < 0,
            mag: trim(vec![x.unsigned_abs()]),
            exp: 0,
        }
    }

    /// Exact value of a finite f64.
    pub fn from_f64(x: f64) -> Self {
        assert!(x.is_finite(), "dyadic of non-finite {x}");
        let bits = x.to_bits();
        let field = ((bits >> 52) & 0x7ff) as i64;
        let frac = bits & ((1 << 52) - 1);
        let (m, exp) = if field == 0 {
            (frac, -1074)
        } else {
            (frac | (1 << 52), field - 1075)
        };
        Dyadic {
            negative: (bits >> 63) != 0 && m != 0,
            mag: trim(vec![m]),
            exp,
        }
    }

    pub fn is_zero(&self) -> bool {
        self.mag.is_empty()
    }

    pub fn neg(&self) -> Self {
        Dyadic {
            negative: !self.negative && !self.is_zero(),
            mag: self.mag.clone(),
            exp: self.exp,
        }
    }

    /// Exact `self * 2^-bits`.
    pub fn scale_down(&self, bits: i64) -> Self {
        Dyadic {
            exp: self.exp - bits,
            ..self.clone()
        }
    }

    pub fn mul(&self, o: &Self) -> Self {
        let mag = mul_mag(&self.mag, &o.mag);
        Dyadic {
            negative: !mag.is_empty() && self.negative != o.negative,
            mag,
            exp: self.exp + o.exp,
        }
    }

    pub fn add(&self, o: &Self) -> Self {
        if self.is_zero() {
            return o.clone();
        }
        if o.is_zero() {
            return self.clone();
        }
        let exp = self.exp.min(o.exp);
        let a = shl_mag(&self.mag, (self.exp - exp) as u64);
        let b = shl_mag(&o.mag, (o.exp - exp) as u64);
        let (negative, mag) = if self.negative == o.negative {
            (self.negative, add_mag(&a, &b))
        } else {
            match cmp_mag(&a, &b) {
                Ordering::Equal => (false, vec![]),
                Ordering::Greater => (self.negative, sub_mag(&a, &b)),
                Ordering::Less => (o.negative, sub_mag(&b, &a)),
            }
        };
        Dyadic { negative, mag, exp }
    }

    pub fn sub(&self, o: &Self) -> Self {
        self.add(&o.neg())
    }

    /// Exact comparison against an f64 bound; `-inf`/`+inf` bound everything finite and NaN
    /// is incomparable (`None`).
    pub fn cmp_f64(&self, bound: f64) -> Option<Ordering> {
        if bound.is_nan() {
            None
        } else if bound == f64::INFINITY {
            Some(Ordering::Less)
        } else if bound == f64::NEG_INFINITY {
            Some(Ordering::Greater)
        } else {
            let d = self.sub(&Dyadic::from_f64(bound));
            Some(if d.is_zero() {
                Ordering::Equal
            } else if d.negative {
                Ordering::Less
            } else {
                Ordering::Greater
            })
        }
    }

    /// `lo <= self <= hi` exactly, with infinite bounds accepted and NaN bounds refused.
    pub fn within(&self, lo: f64, hi: f64) -> bool {
        matches!(self.cmp_f64(lo), Some(Ordering::Greater | Ordering::Equal))
            && matches!(self.cmp_f64(hi), Some(Ordering::Less | Ordering::Equal))
    }
}
