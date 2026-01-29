#!/usr/bin/env python3
"""Parse perf register output to JSON"""
import sys
import json
import re

registers = []
for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    
    # Parse: ip sym AX=... BX=... etc
    parts = line.split()
    if len(parts) < 3:
        continue
    
    entry = {'ip': parts[0], 'sym': parts[1], 'regs': {}}
    
    for part in parts[2:]:
        if '=' in part:
            reg, val = part.split('=', 1)
            try:
                entry['regs'][reg] = int(val, 16)
            except:
                pass
    
    if entry['regs']:
        registers.append(entry)

json.dump(registers, sys.stdout, indent=2)
