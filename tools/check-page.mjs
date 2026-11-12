// Load a page in headless Chrome and report console errors, uncaught exceptions and failed loads,
// including those from Web Workers. Used for each milestone's "page loads with no console errors"
// check. Dependency-free: talks to Chrome over the DevTools protocol with Node's built-in WebSocket
// (Node 22+).
//
// Usage: node tools/check-page.mjs [url] [--until <js expr>] [--eval <js expr>] [--screenshot file.png]
//                                 [--dark] [--timeout seconds]
//   --until       wait until this expression is truthy in the page (default: document load)
//   --eval        print the value of this expression after waiting
//   --eval-file   like --eval, reading the expression from a file
//   --screenshot  save a PNG of the viewport
//   --dark        emulate prefers-color-scheme: dark
//   --timeout     seconds to wait for --until (default 20; long benchmarks need more)
// Exit status 1 if any errors were seen or --until timed out.

import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const CHROME = process.env.CHROME
    ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const DEFAULT_TIMEOUT = 20000; // ms, overall wait for --until
const POLL = 100; // ms between --until checks
const SETTLE = 300; // ms to keep listening after --until succeeds
const VIEWPORT = { width: 1280, height: 900 }; // px

function parseArgs(argv) {
    const opts = { url: 'http://localhost:8000/', until: 'document.readyState === "complete"', timeout: DEFAULT_TIMEOUT };
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (arg === '--until')
            opts.until = argv[++i];
        else if (arg === '--eval')
            opts.eval = argv[++i];
        else if (arg === '--eval-file')
            opts.eval = readFileSync(argv[++i], 'utf8');
        else if (arg === '--screenshot')
            opts.screenshot = argv[++i];
        else if (arg === '--dark')
            opts.dark = true;
        else if (arg === '--timeout')
            opts.timeout = 1000 * Number(argv[++i]);
        else
            opts.url = arg;
    }
    return opts;
}

function launchChrome(profileDir) {
    const proc = spawn(CHROME, [
        '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
        `--user-data-dir=${profileDir}`, '--remote-debugging-port=0',
        `--window-size=${VIEWPORT.width},${VIEWPORT.height}`, 'about:blank',
    ], { stdio: ['ignore', 'ignore', 'pipe'] });
    const wsUrl = new Promise((resolve, reject) => {
        let buf = '';
        proc.stderr.on('data', chunk => {
            buf += chunk;
            const m = buf.match(/DevTools listening on (ws:\/\/\S+)/);
            if (m)
                resolve(m[1]);
        });
        proc.on('exit', code => reject(new Error(`Chrome exited (${code}) before DevTools was ready`)));
    });
    return { proc, wsUrl };
}

// Minimal CDP client over one browser-level WebSocket with flattened sessions.
function connect(wsUrl) {
    const ws = new WebSocket(wsUrl);
    let nextId = 1;
    const pending = new Map();
    const listeners = [];
    ws.addEventListener('message', event => {
        const msg = JSON.parse(event.data);
        if (msg.id && pending.has(msg.id)) {
            const { resolve, reject } = pending.get(msg.id);
            pending.delete(msg.id);
            if (msg.error)
                reject(new Error(`${msg.error.message} (${msg.error.code})`));
            else
                resolve(msg.result);
        }
        else if (msg.method) {
            for (const fn of listeners)
                fn(msg);
        }
    });
    const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
        const id = nextId++;
        pending.set(id, { resolve, reject });
        ws.send(JSON.stringify({ id, method, params, sessionId }));
    });
    const opened = new Promise((resolve, reject) => {
        ws.addEventListener('open', resolve);
        ws.addEventListener('error', reject);
    });
    return { ws, send, onEvent: fn => listeners.push(fn), opened };
}

function describeRemote(arg) {
    if (arg.type === 'string')
        return arg.value;
    if ('value' in arg)
        return JSON.stringify(arg.value);
    return arg.description ?? arg.type;
}

async function main() {
    const opts = parseArgs(process.argv.slice(2));
    const profileDir = mkdtempSync(join(tmpdir(), 'check-page-'));
    const { proc, wsUrl } = launchChrome(profileDir);
    const problems = [];
    const messages = [];
    let exitCode = 0;
    let pageSession = null;
    try {
        const cdp = connect(await wsUrl);
        await cdp.opened;

        // Record console output and errors from the page and any worker attached to it.
        cdp.onEvent(async msg => {
            const where = msg.sessionId === pageSession ? 'page' : 'worker';
            if (msg.method === 'Runtime.consoleAPICalled') {
                const text = msg.params.args.map(describeRemote).join(' ');
                const line = `[${where} console.${msg.params.type}] ${text}`;
                messages.push(line);
                if (msg.params.type === 'error' || msg.params.type === 'assert')
                    problems.push(line);
            }
            else if (msg.method === 'Runtime.exceptionThrown') {
                const d = msg.params.exceptionDetails;
                problems.push(`[${where} exception] ${d.exception?.description ?? d.text} `
                    + `(${d.url ?? ''}:${d.lineNumber + 1})`);
            }
            else if (msg.method === 'Log.entryAdded' && msg.params.entry.level === 'error') {
                const e = msg.params.entry;
                problems.push(`[${where} ${e.source}] ${e.text} ${e.url ?? ''}`);
            }
            else if (msg.method === 'Target.attachedToTarget') {
                const sid = msg.params.sessionId;
                await cdp.send('Runtime.enable', {}, sid).catch(() => {});
                await cdp.send('Log.enable', {}, sid).catch(() => {});
                await cdp.send('Runtime.runIfWaitingForDebugger', {}, sid).catch(() => {});
            }
        });

        const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' });
        ({ sessionId: pageSession } = await cdp.send('Target.attachToTarget',
            { targetId, flatten: true }));
        await cdp.send('Page.enable', {}, pageSession);
        await cdp.send('Runtime.enable', {}, pageSession);
        await cdp.send('Log.enable', {}, pageSession);
        await cdp.send('Emulation.setDeviceMetricsOverride',
            { ...VIEWPORT, deviceScaleFactor: 1, mobile: false }, pageSession);
        if (opts.dark) {
            await cdp.send('Emulation.setEmulatedMedia',
                { features: [{ name: 'prefers-color-scheme', value: 'dark' }] }, pageSession);
        }
        await cdp.send('Target.setAutoAttach',
            { autoAttach: true, waitForDebuggerOnStart: true, flatten: true }, pageSession);
        await cdp.send('Page.navigate', { url: opts.url }, pageSession);

        const evaluate = async expression => {
            const r = await cdp.send('Runtime.evaluate',
                { expression, returnByValue: true, awaitPromise: true }, pageSession);
            if (r.exceptionDetails)
                throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
            return r.result.value;
        };

        const deadline = Date.now() + opts.timeout;
        let ready = false;
        while (Date.now() < deadline) {
            ready = await evaluate(`Boolean(${opts.until})`).catch(() => false);
            if (ready)
                break;
            await new Promise(r => setTimeout(r, POLL));
        }
        if (!ready)
            problems.push(`[timeout] --until not satisfied after ${opts.timeout} ms: ${opts.until}`);
        await new Promise(r => setTimeout(r, SETTLE));

        // Evaluate before printing so console output produced by the expression is reported too.
        const evalResult = opts.eval ? await evaluate(opts.eval) : undefined;
        await new Promise(r => setTimeout(r, SETTLE));
        for (const line of messages)
            console.log(line);
        if (opts.eval)
            console.log(`eval: ${JSON.stringify(evalResult)}`);
        if (opts.screenshot) {
            const { data } = await cdp.send('Page.captureScreenshot', { format: 'png' }, pageSession);
            writeFileSync(opts.screenshot, Buffer.from(data, 'base64'));
            console.log(`screenshot: ${opts.screenshot}`);
        }
        if (problems.length) {
            console.log(`\n${problems.length} problem(s):`);
            for (const p of problems)
                console.log(`  ${p}`);
            exitCode = 1;
        }
        else {
            console.log('OK: no console errors, exceptions or failed loads');
        }
        cdp.ws.close();
    }
    finally {
        // Chrome keeps writing to its profile until it has fully exited.
        // If Chrome already exited (it failed to start), its 'exit' event has passed: waiting for it would leave
        // nothing pending, and Node would quit with status 0 before the error is reported.
        if (proc.exitCode === null && proc.signalCode === null) {
            const exited = new Promise(resolve => proc.once('exit', resolve));
            proc.kill();
            await exited;
        }
        rmSync(profileDir, { recursive: true, force: true, maxRetries: 5 });
    }
    process.exit(exitCode);
}

main().catch(err => {
    console.error(err);
    process.exit(1);
});
