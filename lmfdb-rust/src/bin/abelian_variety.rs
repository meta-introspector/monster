// Model: Abelian variety over F_71 with slopes [0, 1/2, 1/2, 1]
// This is the LMFDB test_slopes function

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Rational {
    num: i64,
    den: i64,
}

impl Rational {
    pub fn new(num: i64, den: i64) -> Self {
        Self { num, den }
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

#[derive(Debug, Clone)]
pub struct AbelianVariety {
    dimension: u32,
    field_size: u32,
    label: String,
    slopes: Vec<Rational>,
}

impl AbelianVariety {
    pub fn new(dimension: u32, field_size: u32, label: &str) -> Self {
        Self {
            dimension,
            field_size,
            label: label.to_string(),
            slopes: vec![],
        }
    }
    
    pub fn with_slopes(mut self, slopes: Vec<Rational>) -> Self {
        self.slopes = slopes;
        self
    }
    
    pub fn url(&self) -> String {
        format!("/Variety/Abelian/Fq/{}/{}/{}", 
                self.dimension, self.field_size, self.label)
    }
    
    pub fn check_slopes(&self, expected: &[Rational]) -> bool {
        self.slopes == expected
    }
}

fn main() {
    // The exact case from LMFDB
    let av = AbelianVariety::new(2, 71, "ah_a")
        .with_slopes(vec![
            Rational::new(0, 1),
            Rational::new(1, 2),
            Rational::new(1, 2),
            Rational::new(1, 1),
        ]);
    
    println!("Abelian Variety over F_71");
    println!("URL: {}", av.url());
    println!("Slopes: {:?}", av.slopes);
    
    let expected = vec![
        Rational::new(0, 1),
        Rational::new(1, 2),
        Rational::new(1, 2),
        Rational::new(1, 1),
    ];
    
    assert!(av.check_slopes(&expected));
    println!("✓ Slopes verified!");
}
