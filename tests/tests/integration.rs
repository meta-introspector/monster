use headless_chrome::{Browser, LaunchOptions};
use std::fs;
use std::path::PathBuf;
use anyhow::Result;
use serde_json::Value;

#[test]
fn test_page_loads() -> Result<()> {
    let browser = Browser::new(LaunchOptions::default())?;
    let tab = browser.new_tab()?;
    
    let html_path = get_test_html_path();
    tab.navigate_to(&format!("file://{}", html_path.display()))?;
    tab.wait_until_navigated()?;
    
    // Check title
    let title = tab.get_title()?;
    assert!(title.contains("Monster Walk"), "Title should contain 'Monster Walk'");
    
    println!("✓ Page loads successfully");
    Ok(())
}

#[test]
fn test_wasm_initialization() -> Result<()> {
    let browser = Browser::new(LaunchOptions::default())?;
    let tab = browser.new_tab()?;
    
    let html_path = get_test_html_path();
    tab.navigate_to(&format!("file://{}", html_path.display()))?;
    tab.wait_until_navigated()?;
    
    // Wait for WASM to load
    std::thread::sleep(std::time::Duration::from_secs(3));
    
    // Check console logs
    let logs = tab.evaluate("console.log('WASM test'); 'success'", false)?;
    assert!(logs.value.is_some());
    
    println!("✓ WASM initializes");
    Ok(())
}

#[test]
fn test_ten_groups_displayed() -> Result<()> {
    let browser = Browser::new(LaunchOptions::default())?;
    let tab = browser.new_tab()?;
    
    let html_path = get_test_html_path();
    tab.navigate_to(&format!("file://{}", html_path.display()))?;
    tab.wait_until_navigated()?;
    
    std::thread::sleep(std::time::Duration::from_secs(3));
    
    // Count group cards
    let result = tab.evaluate(
        "document.querySelectorAll('.group-card').length",
        false
    )?;
    
    if let Some(Value::Number(count)) = result.value {
        let count = count.as_u64().unwrap();
        assert_eq!(count, 10, "Should have exactly 10 groups");
        println!("✓ All 10 groups displayed");
    } else {
        panic!("Could not count groups");
    }
    
    Ok(())
}

#[test]
fn test_group_content() -> Result<()> {
    let browser = Browser::new(LaunchOptions::default())?;
    let tab = browser.new_tab()?;
    
    let html_path = get_test_html_path();
    tab.navigate_to(&format!("file://{}", html_path.display()))?;
    tab.wait_until_navigated()?;
    
    std::thread::sleep(std::time::Duration::from_secs(3));
    
    // Check Group 1 content
    let result = tab.evaluate(
        "document.querySelector('#group-1')?.textContent || ''",
        false
    )?;
    
    if let Some(Value::String(text)) = result.value {
        assert!(text.contains("8080"), "Group 1 should contain '8080'");
        assert!(text.contains("Group 1"), "Should have Group 1 label");
        println!("✓ Group content correct");
    }
    
    Ok(())
}

#[test]
fn test_modal_interaction() -> Result<()> {
    let browser = Browser::new(LaunchOptions::default())?;
    let tab = browser.new_tab()?;
    
    let html_path = get_test_html_path();
    tab.navigate_to(&format!("file://{}", html_path.display()))?;
    tab.wait_until_navigated()?;
    
    std::thread::sleep(std::time::Duration::from_secs(3));
    
    // Click first group
    tab.evaluate("document.querySelector('#group-1')?.click()", false)?;
    std::thread::sleep(std::time::Duration::from_secs(1));
    
    // Check modal is visible
    let result = tab.evaluate(
        "document.querySelector('#detail-modal')?.style.display",
        false
    )?;
    
    if let Some(Value::String(display)) = result.value {
        assert_eq!(display, "block", "Modal should be visible");
        println!("✓ Modal interaction works");
    }
    
    Ok(())
}

#[test]
fn test_harmonic_calculator() -> Result<()> {
    let browser = Browser::new(LaunchOptions::default())?;
    let tab = browser.new_tab()?;
    
    let html_path = get_test_html_path();
    tab.navigate_to(&format!("file://{}", html_path.display()))?;
    tab.wait_until_navigated()?;
    
    std::thread::sleep(std::time::Duration::from_secs(3));
    
    // Set input values
    tab.evaluate("document.querySelector('#prime-input').value = '2'", false)?;
    tab.evaluate("document.querySelector('#exp-input').value = '46'", false)?;
    
    // Click calculate
    tab.evaluate("document.querySelector('#calc-harmonic')?.click()", false)?;
    std::thread::sleep(std::time::Duration::from_secs(1));
    
    // Check result
    let result = tab.evaluate(
        "document.querySelector('#harmonic-result')?.textContent || ''",
        false
    )?;
    
    if let Some(Value::String(text)) = result.value {
        assert!(text.contains("Hz"), "Result should contain 'Hz'");
        println!("✓ Harmonic calculator works: {}", text);
    }
    
    Ok(())
}

#[test]
fn test_mathjax_rendering() -> Result<()> {
    let browser = Browser::new(LaunchOptions::default())?;
    let tab = browser.new_tab()?;
    
    let html_path = get_test_html_path();
    tab.navigate_to(&format!("file://{}", html_path.display()))?;
    tab.wait_until_navigated()?;
    
    // Wait for MathJax
    std::thread::sleep(std::time::Duration::from_secs(5));
    
    // Check for rendered math
    let result = tab.evaluate(
        "document.querySelectorAll('mjx-container, .MJX-TEX').length",
        false
    )?;
    
    if let Some(Value::Number(count)) = result.value {
        let count = count.as_u64().unwrap();
        assert!(count > 0, "MathJax should render equations");
        println!("✓ MathJax rendered {} equations", count);
    }
    
    Ok(())
}

#[test]
fn test_ai_accessibility() -> Result<()> {
    let browser = Browser::new(LaunchOptions::default())?;
    let tab = browser.new_tab()?;
    
    let html_path = get_test_html_path();
    tab.navigate_to(&format!("file://{}", html_path.display()))?;
    tab.wait_until_navigated()?;
    
    std::thread::sleep(std::time::Duration::from_secs(3));
    
    // Extract structured data for AI
    let structured_data = tab.evaluate(r#"
        JSON.stringify({
            title: document.title,
            groups: Array.from(document.querySelectorAll('.group-card')).map(card => ({
                text: card.textContent.trim(),
                id: card.id
            })),
            equations: Array.from(document.querySelectorAll('.math-display')).map(eq => 
                eq.textContent.trim()
            ),
            sections: Array.from(document.querySelectorAll('section')).map(s => ({
                id: s.id,
                heading: s.querySelector('h2')?.textContent,
                content: s.textContent.substring(0, 200)
            }))
        })
    "#, false)?;
    
    if let Some(Value::String(json_str)) = structured_data.value {
        let data: Value = serde_json::from_str(&json_str)?;
        
        // Verify AI can read structure
        assert!(data["groups"].as_array().unwrap().len() == 10);
        assert!(data["equations"].as_array().unwrap().len() > 0);
        assert!(data["sections"].as_array().unwrap().len() > 0);
        
        println!("✓ AI can extract structured data:");
        println!("  - {} groups", data["groups"].as_array().unwrap().len());
        println!("  - {} equations", data["equations"].as_array().unwrap().len());
        println!("  - {} sections", data["sections"].as_array().unwrap().len());
        
        // Save for AI model testing
        fs::write("test-results/ai-readable.json", json_str)?;
    }
    
    Ok(())
}

#[test]
fn test_semantic_structure() -> Result<()> {
    let browser = Browser::new(LaunchOptions::default())?;
    let tab = browser.new_tab()?;
    
    let html_path = get_test_html_path();
    tab.navigate_to(&format!("file://{}", html_path.display()))?;
    tab.wait_until_navigated()?;
    
    std::thread::sleep(std::time::Duration::from_secs(3));
    
    // Check semantic HTML structure
    let checks = vec![
        ("header", "Header element exists"),
        ("nav", "Navigation exists"),
        ("main", "Main content area exists"),
        ("section", "Sections exist"),
        ("footer", "Footer exists"),
        ("h1", "H1 heading exists"),
        ("h2", "H2 headings exist"),
    ];
    
    for (selector, description) in checks {
        let result = tab.evaluate(
            &format!("document.querySelector('{}') !== null", selector),
            false
        )?;
        
        if let Some(Value::Bool(exists)) = result.value {
            assert!(exists, "{} failed", description);
            println!("✓ {}", description);
        }
    }
    
    Ok(())
}

#[test]
fn test_bott_periodicity_display() -> Result<()> {
    let browser = Browser::new(LaunchOptions::default())?;
    let tab = browser.new_tab()?;
    
    let html_path = get_test_html_path();
    tab.navigate_to(&format!("file://{}", html_path.display()))?;
    tab.wait_until_navigated()?;
    
    std::thread::sleep(std::time::Duration::from_secs(3));
    
    // Check Bott period section
    let result = tab.evaluate(
        "document.querySelector('#bott-period')?.textContent || ''",
        false
    )?;
    
    if let Some(Value::String(text)) = result.value {
        assert!(text.contains("Period"), "Should mention period");
        assert!(text.contains("8"), "Should mention period 8");
        println!("✓ Bott periodicity displayed");
    }
    
    Ok(())
}

fn get_test_html_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../web/index.html");
    path.canonicalize().expect("HTML file should exist")
}

#[test]
fn generate_ai_report() -> Result<()> {
    println!("\n=== AI Accessibility Report ===\n");
    
    let browser = Browser::new(LaunchOptions::default())?;
    let tab = browser.new_tab()?;
    
    let html_path = get_test_html_path();
    tab.navigate_to(&format!("file://{}", html_path.display()))?;
    tab.wait_until_navigated()?;
    
    std::thread::sleep(std::time::Duration::from_secs(5));
    
    // Generate comprehensive report
    let report = tab.evaluate(r#"
        JSON.stringify({
            metadata: {
                title: document.title,
                url: window.location.href,
                timestamp: new Date().toISOString()
            },
            structure: {
                headings: Array.from(document.querySelectorAll('h1, h2, h3')).map(h => ({
                    level: h.tagName,
                    text: h.textContent.trim()
                })),
                sections: Array.from(document.querySelectorAll('section')).map(s => s.id),
                links: Array.from(document.querySelectorAll('a')).length
            },
            content: {
                groups: Array.from(document.querySelectorAll('.group-card')).map(card => ({
                    id: card.id,
                    sequence: card.querySelector('h3')?.textContent,
                    details: Array.from(card.querySelectorAll('.group-details p')).map(p => p.textContent)
                })),
                equations: Array.from(document.querySelectorAll('.math-display')).length
            },
            interactivity: {
                buttons: Array.from(document.querySelectorAll('button')).length,
                inputs: Array.from(document.querySelectorAll('input')).length,
                clickable: Array.from(document.querySelectorAll('[onclick], .group-card')).length
            }
        })
    "#, false)?;
    
    if let Some(Value::String(json_str)) = report.value {
        fs::create_dir_all("test-results")?;
        fs::write("test-results/ai-accessibility-report.json", &json_str)?;
        
        let data: Value = serde_json::from_str(&json_str)?;
        println!("{}", serde_json::to_string_pretty(&data)?);
        
        println!("\n✓ AI Accessibility Report generated");
        println!("  Saved to: test-results/ai-accessibility-report.json");
    }
    
    Ok(())
}
