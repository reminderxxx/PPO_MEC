"""Pinned external Ed25519 backend; leaves the scientific Python environment intact."""
from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess

_PROGRAM = r"""
const crypto = require('node:crypto');
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', x => { input += x; });
process.stdin.on('end', () => {
  try {
    const data = JSON.parse(input);
    const raw = Buffer.from(data.public_key, 'base64');
    const signature = Buffer.from(data.signature, 'base64');
    if (raw.length !== 32 || signature.length !== 64) throw new Error('key/signature size');
    const key = crypto.createPublicKey({key: Buffer.concat([
      Buffer.from('302a300506032b6570032100', 'hex'), raw]), format: 'der', type: 'spki'});
    const verified = crypto.verify(null, Buffer.from(data.message, 'base64'), key, signature);
    process.stdout.write(JSON.stringify({verified, node: process.versions.node, openssl: process.versions.openssl}));
  } catch (error) {
    process.stderr.write(String(error)); process.exitCode = 2;
  }
});
"""


def verify(backend, public_key, message, signature):
    if not isinstance(backend,dict) or set(backend)!={'kind','executable','sha256'} or backend['kind']!='node_ed25519_v1':
        raise ValueError('independently pinned signature backend unavailable')
    executable=Path(backend['executable'])
    if not executable.is_absolute() or '..' in executable.parts:
        raise ValueError('signature backend absolute path required')
    if any(path.is_symlink() for path in (executable,*executable.parents)):
        raise ValueError('signature backend symlink')
    digest=hashlib.sha256()
    with executable.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    if digest.hexdigest()!=backend['sha256']:
        raise ValueError('signature backend content drift')
    if len(public_key)!=32 or len(base64.b64decode(signature,validate=True))!=64:return False
    payload={'public_key':base64.b64encode(public_key).decode(),
        'message':base64.b64encode(message).decode(),'signature':signature}
    # No inherited NODE_OPTIONS, NODE_PATH, preload, compile-cache or shell.
    result=subprocess.run([str(executable),'--no-addons','--no-warnings','-e',_PROGRAM],
        input=json.dumps(payload),text=True,capture_output=True,check=False,
        env={'LANG':'C','TZ':'UTC'},timeout=30)
    if result.returncode:raise ValueError('signature backend rejected input: '+result.stderr[:200])
    report=json.loads(result.stdout)
    if set(report)!={'verified','node','openssl'} or type(report['verified']) is not bool:
        raise ValueError('signature backend response schema')
    return report['verified']
