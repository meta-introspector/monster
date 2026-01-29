// Upload ZK memes to Cloudflare KV
const fs = require('fs');
const path = require('path');

const MEMES_DIR = '../zk_memes';
const KV_NAMESPACE = 'ZK_MEMES';

async function uploadMemes() {
  const files = fs.readdirSync(MEMES_DIR).filter(f => f.endsWith('.json'));
  
  console.log(`📤 Uploading ${files.length} ZK memes to Cloudflare KV...`);
  
  for (const file of files) {
    const meme = JSON.parse(fs.readFileSync(path.join(MEMES_DIR, file)));
    const label = meme.label;
    
    // Upload via wrangler CLI
    const cmd = `wrangler kv:key put --binding=${KV_NAMESPACE} "${label}" '${JSON.stringify(meme)}'`;
    console.log(`  ✓ ${label}`);
    
    // Uncomment to actually upload:
    // require('child_process').execSync(cmd);
  }
  
  console.log('✅ Upload complete');
}

uploadMemes();
