#!/bin/zsh
# Usage: cmd.sh '<json command>' [wait_seconds]; then prints status and saves both camera frames.
C=/private/tmp/claude-501/-Users-zhangbocheng-code-projects-research-arm-harness/74ccf756-b267-448e-8051-a944b1ade232/scratchpad/ctl
[ -n "$1" ] && echo "$1" >> $C/cmds.jsonl
sleep 0.3
for i in $(seq 1 ${2:-80}); do
  python3 -c "import json,sys;sys.exit(0 if not json.load(open('$C/status.json'))['moving'] else 1)" 2>/dev/null && break
  sleep 0.1
done
sleep 0.4
curl -s --max-time 3 http://127.0.0.1:8766/frame.jpg -o $C/g.jpg
curl -s --max-time 3 http://127.0.0.1:8765/frame.jpg -o $C/w.jpg
python3 - <<EOF
import json,time
s=json.load(open('$C/status.json'))
print('age %.2fs'%(time.time()-s['t']), 'moving',s['moving'])
for i,m in s['motors'].items(): print(i, 'pos',m['pos'],'tgt',m['target'],'load',m['load'],'T',m['temp'],'V',m['volt'],'err',m.get('err'))
print('last', json.dumps(s['last'])[:300])
EOF
