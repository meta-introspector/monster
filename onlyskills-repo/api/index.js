// onlyskills.com API - Minimal skill registry

const express = require('express');
const app = express();
app.use(express.json());

// In-memory store (replace with PostgreSQL in production)
let skills = require('../onlyskills_profiles.json');

// CORS
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Content-Type');
  next();
});

// GET /api/skills - List all skills
app.get('/api/skills', (req, res) => {
  res.json({
    total: skills.length,
    skills: skills
  });
});

// GET /api/skills/:id - Get skill by ID
app.get('/api/skills/:id', (req, res) => {
  const skill = skills.find(s => s.shard_id === parseInt(req.params.id));
  if (!skill) return res.status(404).json({ error: 'Skill not found' });
  res.json(skill);
});

// GET /api/skills/shard/:id - Get skill by shard
app.get('/api/skills/shard/:id', (req, res) => {
  const skill = skills.find(s => s.shard_id === parseInt(req.params.id));
  if (!skill) return res.status(404).json({ error: 'Shard not found' });
  res.json(skill);
});

// GET /api/skills/type/:type - Get skills by type
app.get('/api/skills/type/:type', (req, res) => {
  const filtered = skills.filter(s => s.search_capability === req.params.type);
  res.json({
    total: filtered.length,
    skills: filtered
  });
});

// POST /api/register - Register new skill
app.post('/api/register', (req, res) => {
  const { skill_name, search_type, command, zkperf_hash } = req.body;
  
  const newSkill = {
    shard_id: skills.length,
    prime: [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71][skills.length % 15],
    skill_name,
    skill_type: `search_${search_type}`,
    command,
    search_capability: search_type,
    zkperf_hash,
    performance: {
      verification_time_ms: 0,
      lines_of_code: 0,
      quantum_amplitude: 1.0 / 71
    }
  };
  
  skills.push(newSkill);
  res.status(201).json(newSkill);
});

// GET /api/stats - Registry statistics
app.get('/api/stats', (req, res) => {
  const stats = {
    total_skills: skills.length,
    total_shards: 71,
    search_types: {
      explicit_search: skills.filter(s => s.search_capability === 'explicit_search').length,
      filter_search: skills.filter(s => s.search_capability === 'filter_search').length,
      find_search: skills.filter(s => s.search_capability === 'find_search').length,
      implicit_search: skills.filter(s => s.search_capability === 'implicit_search').length,
      virtual: skills.filter(s => s.search_capability === 'virtual').length
    },
    platforms: 71,
    quantum_superposition: true
  };
  res.json(stats);
});

// GET / - Root
app.get('/', (req, res) => {
  res.json({
    name: 'onlyskills.com zkERDAProlog Registry',
    version: '1.0.0',
    description: 'Zero-Knowledge 71-Shard Skill Registry for AI Agents',
    endpoints: {
      skills: '/api/skills',
      shard: '/api/skills/shard/:id',
      type: '/api/skills/type/:type',
      register: '/api/register',
      stats: '/api/stats'
    },
    platforms: 71,
    shards: 71
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🌐 onlyskills.com API running on port ${PORT}`);
  console.log(`📊 ${skills.length} skills registered`);
  console.log(`🔢 71 shards across 71 platforms`);
});

module.exports = app;
