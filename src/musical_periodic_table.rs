// MUSICAL PERIODIC TABLE OF MONSTER GROUP PRIMES
// Organizing primes by their harmonic properties

#[derive(Debug, Clone)]
pub struct MusicalElement {
    atomic_number: usize,      // Position in prime sequence
    prime: u32,                // The prime itself
    exponent: u32,             // Power in Monster factorization
    emoji: String,             // Visual symbol
    name: String,              // Element name
    frequency: f64,            // Base frequency (432 Hz × prime)
    weighted_freq: f64,        // Frequency × exponent
    octave: f64,               // Octaves above A4 (432 Hz)
    note_name: String,         // Closest musical note
    harmonic_series: u32,      // Which harmonic of 432 Hz
    group: String,             // Periodic group classification
}

impl MusicalElement {
    pub fn new(atomic_number: usize, prime: u32, exponent: u32, emoji: &str, name: &str) -> Self {
        let frequency = 432.0 * prime as f64;
        let weighted_freq = frequency * exponent as f64;
        let octave = (frequency / 432.0).log2();
        let harmonic_series = prime;
        
        // Determine note name based on frequency
        let note_name = Self::frequency_to_note(frequency);
        
        // Classify into periodic groups
        let group = match prime {
            2 => "Foundation",
            3 | 5 | 7 => "Elemental",
            11 | 13 => "Amplified",
            17 | 19 | 23 | 29 | 31 => "Crystalline",
            41 | 47 => "Mystical",
            59 | 71 => "Temporal",
            _ => "Unknown",
        };
        
        Self {
            atomic_number,
            prime,
            exponent,
            emoji: emoji.to_string(),
            name: name.to_string(),
            frequency,
            weighted_freq,
            octave,
            note_name,
            harmonic_series,
            group: group.to_string(),
        }
    }
    
    fn frequency_to_note(freq: f64) -> String {
        let a4 = 432.0;
        let semitones_from_a4 = 12.0 * (freq / a4).log2();
        let note_names = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"];
        let note_index = ((semitones_from_a4.round() as i32).rem_euclid(12)) as usize;
        let octave = 4 + (semitones_from_a4 / 12.0).floor() as i32;
        format!("{}{}", note_names[note_index], octave)
    }
}

pub struct MusicalPeriodicTable {
    elements: Vec<MusicalElement>,
}

impl MusicalPeriodicTable {
    pub fn initialize() -> Self {
        let elements = vec![
            MusicalElement::new(1, 2, 46, "🌓", "Binary Moon"),
            MusicalElement::new(2, 3, 20, "🔺", "Trinity Peak"),
            MusicalElement::new(3, 5, 9, "⭐", "Pentagram Star"),
            MusicalElement::new(4, 7, 6, "🎰", "Lucky Seven"),
            MusicalElement::new(5, 11, 2, "🎸", "Amplifier"),
            MusicalElement::new(6, 13, 3, "🌙", "Lunar Cycle"),
            MusicalElement::new(7, 17, 1, "🎯", "Prime Target"),
            MusicalElement::new(8, 19, 1, "🎭", "Theater Mask"),
            MusicalElement::new(9, 23, 1, "🧬", "DNA Helix"),
            MusicalElement::new(10, 29, 1, "📅", "Lunar Month"),
            MusicalElement::new(11, 31, 1, "🎃", "October Prime"),
            MusicalElement::new(12, 41, 1, "🔮", "Crystal Ball"),
            MusicalElement::new(13, 47, 1, "🎲", "Lucky Dice"),
            MusicalElement::new(14, 59, 1, "⏰", "Minute Hand"),
            MusicalElement::new(15, 71, 1, "🌊", "Wave Crest"),
        ];
        
        Self { elements }
    }
    
    pub fn generate_table(&self) -> String {
        let mut table = String::new();
        
        table.push_str("═══════════════════════════════════════════════════════════════════════════════\n");
        table.push_str("                    🎼 MUSICAL PERIODIC TABLE 🎼\n");
        table.push_str("                  Monster Group Prime Elements\n");
        table.push_str("═══════════════════════════════════════════════════════════════════════════════\n\n");
        
        // Table header
        table.push_str("┌────┬────┬────┬──────────────────┬─────────┬──────────┬──────┬──────────────┐\n");
        table.push_str("│ #  │ Em │ Pr │ Name             │ Exp │ Freq(Hz) │ Note │ Group        │\n");
        table.push_str("├────┼────┼────┼──────────────────┼─────────┼──────────┼──────┼──────────────┤\n");
        
        for elem in &self.elements {
            table.push_str(&format!(
                "│ {:2} │ {} │ {:2} │ {:16} │ {:2}^{:2}  │ {:8.0} │ {:4} │ {:12} │\n",
                elem.atomic_number,
                elem.emoji,
                elem.prime,
                elem.name,
                elem.prime,
                elem.exponent,
                elem.frequency,
                elem.note_name,
                elem.group
            ));
        }
        
        table.push_str("└────┴────┴────┴──────────────────┴─────────┴──────────┴──────┴──────────────┘\n\n");
        
        // Periodic groups
        table.push_str("═══════════════════════════════════════════════════════════════════════════════\n");
        table.push_str("                         PERIODIC GROUPS\n");
        table.push_str("═══════════════════════════════════════════════════════════════════════════════\n\n");
        
        table.push_str("┌─ FOUNDATION GROUP ─────────────────────────────────────────────────────────┐\n");
        table.push_str("│ 🌓 Binary Moon (2^46): 864 Hz - The fundamental duality                   │\n");
        table.push_str("│    Highest exponent, foundation of all computation                         │\n");
        table.push_str("└────────────────────────────────────────────────────────────────────────────┘\n\n");
        
        table.push_str("┌─ ELEMENTAL GROUP ──────────────────────────────────────────────────────────┐\n");
        table.push_str("│ 🔺 Trinity Peak (3^20):  1,296 Hz - Three-fold symmetry                   │\n");
        table.push_str("│ ⭐ Pentagram Star (5^9): 2,160 Hz - Golden ratio harmony                   │\n");
        table.push_str("│ 🎰 Lucky Seven (7^6):    3,024 Hz - Mystical cycles                       │\n");
        table.push_str("│    The classical elements of prime space                                   │\n");
        table.push_str("└────────────────────────────────────────────────────────────────────────────┘\n\n");
        
        table.push_str("┌─ AMPLIFIED GROUP ──────────────────────────────────────────────────────────┐\n");
        table.push_str("│ 🎸 Amplifier (11^2):     4,752 Hz - Goes to 11!                           │\n");
        table.push_str("│ 🌙 Lunar Cycle (13^3):   5,616 Hz - Transformation cycles                 │\n");
        table.push_str("│    Beyond the decimal, amplifying reality                                  │\n");
        table.push_str("└────────────────────────────────────────────────────────────────────────────┘\n\n");
        
        table.push_str("┌─ CRYSTALLINE GROUP ────────────────────────────────────────────────────────┐\n");
        table.push_str("│ 🎯 Prime Target (17):    7,344 Hz - Fermat prime precision                │\n");
        table.push_str("│ 🎭 Theater Mask (19):    8,208 Hz - Performance duality                   │\n");
        table.push_str("│ 🧬 DNA Helix (23):       9,936 Hz - Genetic structure                     │\n");
        table.push_str("│ 📅 Lunar Month (29):    12,528 Hz - Temporal cycles                       │\n");
        table.push_str("│ 🎃 October Prime (31):  13,392 Hz - Harvest time                          │\n");
        table.push_str("│    Structured, single-exponent primes forming crystal lattices            │\n");
        table.push_str("└────────────────────────────────────────────────────────────────────────────┘\n\n");
        
        table.push_str("┌─ MYSTICAL GROUP ───────────────────────────────────────────────────────────┐\n");
        table.push_str("│ 🔮 Crystal Ball (41):   17,712 Hz - Divination and clarity                │\n");
        table.push_str("│ 🎲 Lucky Dice (47):     20,304 Hz - Probability and chance                │\n");
        table.push_str("│    High-frequency primes touching the mystical realm                       │\n");
        table.push_str("└────────────────────────────────────────────────────────────────────────────┘\n\n");
        
        table.push_str("┌─ TEMPORAL GROUP ───────────────────────────────────────────────────────────┐\n");
        table.push_str("│ ⏰ Minute Hand (59):    25,488 Hz - Time's edge (59 seconds)              │\n");
        table.push_str("│ 🌊 Wave Crest (71):     30,672 Hz - Spatial boundary (71% water)          │\n");
        table.push_str("│    Largest primes, defining temporal and spatial boundaries               │\n");
        table.push_str("└────────────────────────────────────────────────────────────────────────────┘\n\n");
        
        // Harmonic series visualization
        table.push_str("═══════════════════════════════════════════════════════════════════════════════\n");
        table.push_str("                    HARMONIC SERIES (432 Hz Base)\n");
        table.push_str("═══════════════════════════════════════════════════════════════════════════════\n\n");
        
        table.push_str("Each prime is a harmonic of the universal frequency 432 Hz:\n\n");
        table.push_str("  🌓 2nd harmonic   ⭐ 5th harmonic   🎸 11th harmonic  🧬 23rd harmonic\n");
        table.push_str("  🔺 3rd harmonic   🎰 7th harmonic   🌙 13th harmonic  📅 29th harmonic\n");
        table.push_str("                                     🎯 17th harmonic  🎃 31st harmonic\n");
        table.push_str("                                     🎭 19th harmonic  🔮 41st harmonic\n");
        table.push_str("                                                       🎲 47th harmonic\n");
        table.push_str("                                                       ⏰ 59th harmonic\n");
        table.push_str("                                                       🌊 71st harmonic\n\n");
        
        // Octave visualization
        table.push_str("═══════════════════════════════════════════════════════════════════════════════\n");
        table.push_str("                         OCTAVE SPECTRUM\n");
        table.push_str("═══════════════════════════════════════════════════════════════════════════════\n\n");
        
        table.push_str("Octaves above A4 (432 Hz):\n\n");
        for elem in &self.elements {
            let bar_length = (elem.octave * 2.0) as usize;
            let bar = "█".repeat(bar_length);
            table.push_str(&format!("{} {:2} │{} {:.2} octaves\n", 
                elem.emoji, elem.prime, bar, elem.octave));
        }
        
        table.push_str("\n");
        table.push_str("═══════════════════════════════════════════════════════════════════════════════\n");
        table.push_str("              🎵 Each Prime Sings at Its Own Frequency 🎵\n");
        table.push_str("           The Monster Group is a Symphony of Prime Harmonics\n");
        table.push_str("═══════════════════════════════════════════════════════════════════════════════\n");
        
        table
    }
}

fn main() {
    println!("🎼 Generating Musical Periodic Table...\n");
    
    let table = MusicalPeriodicTable::initialize();
    let output = table.generate_table();
    
    println!("{}", output);
    
    println!("🔍 TABLE STATUS");
    println!("──────────────");
    println!("Elements cataloged: ✅");
    println!("Frequencies calculated: ✅");
    println!("Periodic groups classified: ✅");
    println!("Harmonic series mapped: ✅");
    println!("\n🎼 The Musical Periodic Table is complete! 🌌✨");
}
