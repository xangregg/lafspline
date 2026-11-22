// tools/serve.py, the dev server behind npm run serve and the page checks. (Regression: python3 -m http.server's
// backlog of 5 reset connections when the page's modules were requested at once on a busy machine, so the app
// never started and check:ui failed now and then.)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import http from 'node:http';

const PORT = 8092; // away from 8000–8002, which the related projects' servers use
const ROUNDS = 50; // bursts
const WIDTH = 60; // simultaneous requests per burst; python3 -m http.server reset about a third of them
const START_TIMEOUT = 5000; // ms
const PATHS = ['/index.html', '/styles.css', '/src/ui/main.js', '/src/core/fit.js'];

const hasPython = spawnSync('python3', ['--version']).status === 0;

function get(path) {
    return new Promise(resolve => {
        const req = http.get({ host: '127.0.0.1', port: PORT, path, agent: false }, res => {
            res.resume();
            res.on('end', () => resolve(res.statusCode));
        });
        req.on('error', e => resolve(e.code));
    });
}

async function waitForServer() {
    const deadline = Date.now() + START_TIMEOUT;
    while (Date.now() < deadline) {
        if (await get('/index.html') === 200)
            return;
        await new Promise(r => setTimeout(r, 100));
    }
    throw new Error('tools/serve.py did not start');
}

test('serve.py: bursts of simultaneous requests are all answered', { skip: !hasPython && 'no python3' }, async () => {
    const server = spawn('python3', ['tools/serve.py', String(PORT)], { stdio: 'ignore' });
    try {
        await waitForServer();
        const outcomes = {};
        for (let r = 0; r < ROUNDS; r++) {
            const codes = await Promise.all(Array.from({ length: WIDTH }, (_, i) => get(PATHS[i % PATHS.length])));
            for (const c of codes)
                outcomes[c] = (outcomes[c] ?? 0) + 1;
        }
        assert.deepEqual(outcomes, { 200: ROUNDS * WIDTH });
    }
    finally {
        server.kill();
    }
});
