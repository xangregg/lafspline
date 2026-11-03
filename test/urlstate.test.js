import { test } from 'node:test';
import assert from 'node:assert/strict';
import { stateToHash, hashToState, packText, unpackText, signalDefaults, MAX_N } from '../src/ui/urlstate.js';
import { SIGNALS } from '../src/core/signals.js';

// Mirrors the initial state in src/ui/main.js.
const DEFAULTS = {
    source: 'ramp', n: SIGNALS.ramp.n, sigma: SIGNALS.ramp.sigma, seed: 1, showTruth: true, userText: '',
    xCol: null, yCol: null, region: 'band', family: 'blocks', calibrate: true, tau: 3, boxes: 'tightest',
    alpha: 0.95, covariance: 'bayes', lambdaMode: 'gcv', logLambda: 0, undersmooth: 1 / 16,
    nseg: null, diff: 2, penalty: 'quadratic', ambition: 0, flat: 0, logKappa: 4, logEpsilon: -2,
};

test('defaults give an empty hash and an empty hash gives the defaults', async () => {
    assert.deepEqual(await stateToHash(DEFAULTS, DEFAULTS), { hash: '', dataOmitted: 0 });
    const { state, rejected } = await hashToState('', DEFAULTS);
    assert.deepEqual(state, DEFAULTS);
    assert.deepEqual(rejected, []);
});

test('changed settings round-trip and only they are written', async () => {
    const st = {
        ...DEFAULTS, source: 'sine', n: 300, sigma: SIGNALS.sine.sigma, seed: 42, alpha: 0.9, covariance: 'sandwich',
        undersmooth: 1 / 4, nseg: 25, diff: 3, penalty: 'sparse', ambition: -2.5, flat: 0.3, logKappa: 6,
        logEpsilon: -3,
    };
    const { hash } = await stateToHash(st, DEFAULTS);
    // σ is the sine's own default, so it is not written
    assert.equal(hash, '#source=sine&n=300&seed=42&alpha=0.9&covariance=sandwich&undersmooth=4&nseg=25&diff=3'
        + '&penalty=sparse&ambition=-2.5&flat=0.3&logKappa=6&logEpsilon=-3');
    const back = await hashToState(hash, DEFAULTS);
    assert.deepEqual(back.state, st);
    assert.deepEqual(back.rejected, []);
});

test('Aₙ settings round-trip', async () => {
    const st = { ...DEFAULTS, region: 'An', family: 'quarter', calibrate: false, tau: 2.5, boxes: 'all' };
    const { hash } = await stateToHash(st, DEFAULTS);
    assert.equal(hash, '#region=An&family=quarter&calibrate=0&tau=2.5&boxes=all');
    const back = await hashToState(hash, DEFAULTS);
    assert.deepEqual(back.state, st);
    assert.deepEqual((await hashToState('#region=Bn&family=some&tau=0&boxes=x', DEFAULTS)).rejected.sort(),
        ['boxes', 'family', 'region', 'tau']);
});

test('a given λ and no undersmoothing round-trip; auto segments are not written', async () => {
    const st = { ...DEFAULTS, lambdaMode: 'given', logLambda: 3.5, undersmooth: 1 };
    const { hash } = await stateToHash(st, DEFAULTS);
    assert.equal(hash, '#lambdaMode=given&logLambda=3.5&undersmooth=1');
    assert.deepEqual((await hashToState(hash, DEFAULTS)).state, st);
    assert.equal((await hashToState('#nseg=auto', DEFAULTS)).state.nseg, null);
});

test('choosing a signal brings its own σ and n', async () => {
    const { state } = await hashToState('#source=exp5', DEFAULTS);
    assert.equal(state.n, Math.min(SIGNALS.exp5.n, MAX_N));
    assert.equal(state.sigma, SIGNALS.exp5.sigma);
    assert.deepEqual(signalDefaults('blocks'), { n: Math.min(SIGNALS.blocks.n, MAX_N), sigma: SIGNALS.blocks.sigma });
    assert.equal(signalDefaults('user'), null);
});

test('invalid and unknown values are ignored and reported', async () => {
    const { state, rejected } = await hashToState(
        '#source=nope&n=1&alpha=2&covariance=hc3&undersmooth=3&nseg=2&diff=4&ambition=9&flat=-0.1&logKappa=0'
        + '&logEpsilon=x&zzz=1&seed=7&penalty=l1',
        DEFAULTS);
    assert.deepEqual(state, { ...DEFAULTS, seed: 7 });
    assert.deepEqual(rejected.sort(), ['alpha', 'ambition', 'covariance', 'diff', 'flat', 'logEpsilon', 'logKappa', 'n',
        'nseg', 'penalty', 'source', 'undersmooth', 'zzz']);
});

test('pasted data round-trips compressed; simulated-only keys are not written for user data', async () => {
    const text = 'age,correct\n' + Array.from({ length: 100 }, (_, i) => `${17 + (i % 30)},${(i * 7) % 13}`).join('\n');
    const st = { ...DEFAULTS, source: 'user', userText: text, xCol: 0, yCol: 1, seed: 9 };
    const { hash, dataOmitted } = await stateToHash(st, DEFAULTS);
    assert.equal(dataOmitted, 0);
    assert.ok(!hash.includes('seed='));
    assert.ok(hash.length < text.length, `compressed hash ${hash.length} vs text ${text.length}`);
    const back = await hashToState(hash, DEFAULTS);
    assert.equal(back.state.userText, text);
    assert.equal(back.state.xCol, 0);
    assert.equal(back.state.yCol, 1);
});

test('data too long for a link is left out and reported', async () => {
    let seed = 1;
    const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const text = Array.from({ length: 4000 }, () => `${rand()},${rand()}`).join('\n');
    const { hash, dataOmitted } = await stateToHash({ ...DEFAULTS, source: 'user', userText: text }, DEFAULTS);
    assert.equal(dataOmitted, text.length);
    assert.ok(!hash.includes('data='));
});

test('packText/unpackText round-trip Unicode; malformed data is rejected', async () => {
    const s = 'x,y\n1,σ̂ = 3.145 → Aₙ\n';
    assert.equal(await unpackText(await packText(s)), s);
    const { rejected } = await hashToState('#source=user&data=!!notbase64', DEFAULTS);
    assert.deepEqual(rejected, ['data']);
});

test('bundled examples are kept in links by id; simulation settings are not written for them', async () => {
    const st = { ...DEFAULTS, source: 'example:nile-flow', seed: 7, ambition: -1 };
    const { hash } = await stateToHash(st, DEFAULTS);
    assert.equal(hash, '#source=example%3Anile-flow&ambition=-1');
    const back = await hashToState(hash, DEFAULTS);
    assert.equal(back.state.source, 'example:nile-flow');
    assert.deepEqual(back.rejected, []);
    for (const bad of ['example:', 'example:Nile', 'example:a b', 'example:../x'])
        assert.deepEqual((await hashToState(`#source=${encodeURIComponent(bad)}`, DEFAULTS)).rejected, ['source'], bad);
});
