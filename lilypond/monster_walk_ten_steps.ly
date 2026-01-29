\version "2.24.0"

\header {
  title = "The Monster Walk: Ten Steps Down to Earth"
  subtitle = "8080 in All Forms"
  composer = "Generated from Monster Group Theory"
  tagline = "Proven in Lean4, Rust, Prolog, MiniZinc, and more"
}

% Monster primes as frequencies (Hz)
% Base frequency: A4 = 440 Hz
% Each prime maps to a frequency: 440 * (prime / 71)

monsterWalkMelody = \relative c' {
  \clef treble
  \key c \major
  \time 8/8
  \tempo "Andante" 4 = 80
  
  % STEP 1: Lean4 (Formal Proof)
  % Prime 2 (binary): 440 * (2/71) ≈ 12.4 Hz → C1
  c,1~ | c2 r2 |
  ^\markup { \bold "Step 1: Lean4 Formal Proof" }
  ^\markup { \italic "monster_starts_with_8080" }
  
  % STEP 2: Rust (Computation)
  % Prime 3: 440 * (3/71) ≈ 18.6 Hz → D1
  d,1~ | d2 r2 |
  ^\markup { \bold "Step 2: Rust Computation" }
  ^\markup { \italic "8080 verified" }
  
  % STEP 3: Prolog (Logic)
  % Prime 5: 440 * (5/71) ≈ 31.0 Hz → G1
  g,1~ | g2 r2 |
  ^\markup { \bold "Step 3: Prolog Logic" }
  ^\markup { \italic "monster_walk(8080)" }
  
  % STEP 4: MiniZinc (Constraints)
  % Prime 7: 440 * (7/71) ≈ 43.4 Hz → A1
  a,1~ | a2 r2 |
  ^\markup { \bold "Step 4: MiniZinc Constraints" }
  ^\markup { \italic "minimize digits" }
  
  % STEP 5: Song (Lyrics)
  % Prime 11: 440 * (11/71) ≈ 68.2 Hz → C2
  c1~ | c2 r2 |
  ^\markup { \bold "Step 5: Song with Lyrics" }
  ^\markup { \italic "In every base we go!" }
  
  % STEP 6: Picture (HTML)
  % Prime 13: 440 * (13/71) ≈ 80.6 Hz → D2
  d1~ | d2 r2 |
  ^\markup { \bold "Step 6: Picture (HTML)" }
  ^\markup { \italic "Visual proof" }
  
  % STEP 7: NFT (Metadata)
  % Prime 17: 440 * (17/71) ≈ 105.4 Hz → G2
  g1~ | g2 r2 |
  ^\markup { \bold "Step 7: NFT Metadata" }
  ^\markup { \italic "On-chain proof" }
  
  % STEP 8: Meme (Markdown)
  % Prime 19: 440 * (19/71) ≈ 117.7 Hz → A2
  a1~ | a2 r2 |
  ^\markup { \bold "Step 8: Meme (Markdown)" }
  ^\markup { \italic "Viral mathematics" }
  
  % STEP 9: Hexadecimal (Base 16)
  % Prime 23: 440 * (23/71) ≈ 142.5 Hz → C3
  c'1~ | c'2 r2 |
  ^\markup { \bold "Step 9: Hexadecimal" }
  ^\markup { \italic "0x1F90" }
  
  % STEP 10: All Bases (2-71)
  % Prime 71: 440 * (71/71) = 440 Hz → A4
  a'1~ | a'2 r2 |
  ^\markup { \bold "Step 10: All Bases" }
  ^\markup { \italic "Base 71 minimal!" }
  
  \bar "|."
}

monsterWalkChords = \chordmode {
  % Step 1-2: Binary/Ternary (simple)
  c1:maj | c2:maj r2 |
  d1:min | d2:min r2 |
  
  % Step 3-4: Quinary/Septenary (building)
  g1:maj | g2:maj r2 |
  a1:min | a2:min r2 |
  
  % Step 5-6: Undecimal/Tridecimal (middle)
  c1:maj7 | c2:maj7 r2 |
  d1:min7 | d2:min7 r2 |
  
  % Step 7-8: Heptadecimal/Nonadecimal (complex)
  g1:maj9 | g2:maj9 r2 |
  a1:min9 | a2:min9 r2 |
  
  % Step 9-10: Trivigesimal/Unseptuagesimal (resolution)
  c1:maj13 | c2:maj13 r2 |
  a1:maj | a2:maj r2 |
}

monsterWalkBass = \relative c, {
  \clef bass
  \key c \major
  \time 8/8
  
  % Group 1: 8 factors (bass line)
  % 7, 11, 17, 19, 29, 31, 41, 59
  
  c1~ | c2 r2 |  % Step 1
  d1~ | d2 r2 |  % Step 2
  g,1~ | g,2 r2 | % Step 3
  a,1~ | a,2 r2 | % Step 4
  c1~ | c2 r2 |  % Step 5
  d1~ | d2 r2 |  % Step 6
  g,1~ | g,2 r2 | % Step 7
  a,1~ | a,2 r2 | % Step 8
  c1~ | c2 r2 |  % Step 9
  a,1~ | a,2 r2 | % Step 10
}

monsterWalkLyrics = \lyricmode {
  % Step 1
  Lean4 proves the walk is real
  
  % Step 2
  Rust com -- putes with speed and zeal
  
  % Step 3
  Pro -- log rea -- sons through the night
  
  % Step 4
  Mi -- ni -- Zinc finds what is right
  
  % Step 5
  Songs we sing in ev -- ery base
  
  % Step 6
  Pic -- tures show the Mon -- ster's face
  
  % Step 7
  N -- F -- Ts on block -- chain stored
  
  % Step 8
  Memes spread wide, the truth re -- stored
  
  % Step 9
  Hex -- a -- dec -- i -- mal so clean
  
  % Step 10
  Sev -- en -- ty -- one, the fi -- nal scene!
}

\score {
  <<
    \new ChordNames \monsterWalkChords
    \new Staff \with {
      instrumentName = "Melody"
    } {
      \monsterWalkMelody
    }
    \new Lyrics \lyricsto "monsterWalkMelody" {
      \monsterWalkLyrics
    }
    \new Staff \with {
      instrumentName = "Bass"
    } {
      \monsterWalkBass
    }
  >>
  \layout {
    \context {
      \Score
      \override SpacingSpanner.base-shortest-duration = #(ly:make-moment 1/8)
    }
  }
  \midi {
    \tempo 4 = 80
  }
}

% Additional movements for each base
\markup {
  \column {
    \line { \bold "Movement I: Binary (Base 2)" }
    \line { "13 digits: 1111110010000" }
    \line { "Frequency: 12.4 Hz (C1)" }
    \line { "" }
    \line { \bold "Movement II: Octal (Base 8)" }
    \line { "5 digits: 17620" }
    \line { "Frequency: 49.6 Hz (G1)" }
    \line { "" }
    \line { \bold "Movement III: Decimal (Base 10)" }
    \line { "4 digits: 8080" }
    \line { "Frequency: 62.0 Hz (B1)" }
    \line { "" }
    \line { \bold "Movement IV: Hexadecimal (Base 16)" }
    \line { "4 digits: 1F90" }
    \line { "Frequency: 99.2 Hz (G2)" }
    \line { "" }
    \line { \bold "Movement V: Base 71 (Minimal)" }
    \line { "2 digits: 1m (113×71 + 57)" }
    \line { "Frequency: 440 Hz (A4)" }
  }
}

% Frequency table for all 15 Monster primes
\markup {
  \column {
    \line { \bold "Monster Prime Frequencies" }
    \line { "Base A4 = 440 Hz, scaled by (prime / 71)" }
    \line { "" }
    \line { "2:  12.4 Hz  (C1)" }
    \line { "3:  18.6 Hz  (D1)" }
    \line { "5:  31.0 Hz  (G1)" }
    \line { "7:  43.4 Hz  (A1)" }
    \line { "11: 68.2 Hz  (C2)" }
    \line { "13: 80.6 Hz  (D2)" }
    \line { "17: 105.4 Hz (G2)" }
    \line { "19: 117.7 Hz (A2)" }
    \line { "23: 142.5 Hz (C3)" }
    \line { "29: 179.7 Hz (F3)" }
    \line { "31: 192.1 Hz (G3)" }
    \line { "41: 254.1 Hz (C4)" }
    \line { "47: 291.5 Hz (D4)" }
    \line { "59: 365.6 Hz (F#4)" }
    \line { "71: 440.0 Hz (A4)" }
  }
}
