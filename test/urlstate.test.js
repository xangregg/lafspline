import { test } from 'node:test';
import assert from 'node:assert/strict';
import { stateToHash, hashToState, packText, unpackText, signalDefaults, MAX_N } from '../src/ui/urlstate.js';
import { SIGNALS } from '../src/core/signals.js';

// Mirrors the initial state in src/ui/main.js.
const DEFAULTS = {
    source: 'ramp', n: SIGNALS.ramp.n, sigma: SIGNALS.ramp.sigma, seed: 1, showTruth: true, userText: '',
    xCol: null, yCol: null, region: 'band', family: 'blocks', calibrate: true, tau: 3, boxes: 'tightest',
    alpha: 0.95, covariance: 'bayes', bootWeights: 'matched', lambdaMode: 'gcv', logLambda: 0, undersmooth: 1 / 16,
    nseg: null, diff: 2, penalty: 'quadratic', ambition: 0, flat: 0, bendLength: 0.05, extremumCost: 2,
    inflectionCost: 1, logKappa: 4, logEpsilon: -2, showData: true, showCenter: true, showBand: true, showFit: true,
    featureScale: 0.1, view: 'single', panes: ['example:nile-flow', 'sine', 'box', 'step', 'example:ozone', 'ramp'],
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

test('display toggles round-trip', async () => {
    const st = { ...DEFAULTS, showCenter: false, showBand: false };
    const { hash } = await stateToHash(st, DEFAULTS);
    assert.equal(hash, '#showCenter=0&showBand=0');
    assert.deepEqual((await hashToState(hash, DEFAULTS)).state, st);
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

test('bootstrap band settings round-trip; unknown weights are rejected', async () => {
    const st = { ...DEFAULTS, region: 'boot', bootWeights: 'dirichlet' };
    const { hash } = await stateToHash(st, DEFAULTS);
    assert.equal(hash, '#region=boot&bootWeights=dirichlet');
    assert.deepEqual((await hashToState(hash, DEFAULTS)).state, st);
    assert.deepEqual((await hashToState('#bootWeights=poisson', DEFAULTS)).rejected, ['bootWeights']);
});

test('rounded bends round-trip; a bend length or cost out of range is rejected', async () => {
    const st = { ...DEFAULTS, penalty: 'shaped', bendLength: 0.08, extremumCost: 0, inflectionCost: 3.5 };
    const { hash } = await stateToHash(st, DEFAULTS);
    assert.equal(hash, '#penalty=shaped&bendLength=0.08&extremumCost=0&inflectionCost=3.5');
    assert.deepEqual((await hashToState(hash, DEFAULTS)).state, st);
    assert.deepEqual((await hashToState('#bendLength=0.5&extremumCost=-1', DEFAULTS)).rejected,
        ['bendLength', 'extremumCost']);
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
    assert.deepEqual(signalDefaults('heavisine'),
        { n: Math.min(SIGNALS.heavisine.n, MAX_N), sigma: SIGNALS.heavisine.sigma });
    assert.equal(signalDefaults('user'), null);
    // signals taken out of the menu (Blocks, Bumps, Doppler): an old link falls back to the default source
    const old = await hashToState('#source=blocks', DEFAULTS);
    assert.equal(old.state.source, DEFAULTS.source);
    assert.deepEqual(old.rejected, ['source']);
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
    for (const bad of ['example:', 'example:Nile', 'example:a b', 'example:../x', 'local:../x', 'other:x'])
        assert.deepEqual((await hashToState(`#source=${encodeURIComponent(bad)}`, DEFAULTS)).rejected, ['source'], bad);
});

test('local data sets are kept in links by id, as bundled ones are', async () => {
    const st = { ...DEFAULTS, source: 'local:kyoto-cherry-blossom' };
    const { hash } = await stateToHash(st, DEFAULTS);
    assert.equal(hash, '#source=local%3Akyoto-cherry-blossom');
    assert.equal((await hashToState(hash, DEFAULTS)).state.source, 'local:kyoto-cherry-blossom');
});

test('the gallery round-trips with its panes; the panes stay out of a single-view link', async () => {
    const panes = ['example:nile-flow', 'local:kyoto-cherry-blossom', 'heavisine', 'step'];
    const st = { ...DEFAULTS, source: 'example:ozone', view: 'gallery', panes, seed: 7 };
    const { hash } = await stateToHash(st, DEFAULTS);
    // the seed is written for the gallery's simulated panes although the single view's source is a data file
    assert.equal(hash, `#source=example%3Aozone&view=gallery&panes=${encodeURIComponent(panes.join(','))}&seed=7`);
    const back = await hashToState(hash, DEFAULTS);
    assert.deepEqual(back.state, st);
    assert.deepEqual(back.rejected, []);
    assert.equal((await stateToHash({ ...st, view: 'single' }, DEFAULTS)).hash, '#source=example%3Aozone');
    // the default panes, compared as written, are left out
    assert.equal((await stateToHash({ ...DEFAULTS, view: 'gallery', panes: [...DEFAULTS.panes] }, DEFAULTS)).hash,
        '#view=gallery');
});

test('a pane list with pasted data, an unknown source or an unsupported count is rejected', async () => {
    for (const list of ['sine,user,box,step', 'sine,nosuch,box,step', 'sine,box,step', 'sine,box,step,ramp,line']) {
        const { state, rejected } = await hashToState(`#view=gallery&panes=${encodeURIComponent(list)}`, DEFAULTS);
        assert.deepEqual(rejected, ['panes'], list);
        assert.deepEqual(state.panes, DEFAULTS.panes, list);
        assert.equal(state.view, 'gallery');
    }
    assert.deepEqual((await hashToState('#view=grid', DEFAULTS)).rejected, ['view']);
});

test('λ by feature scale round-trips; a scale out of range is rejected', async () => {
    const st = { ...DEFAULTS, lambdaMode: 'scale', featureScale: 0.15 };
    const { hash } = await stateToHash(st, DEFAULTS);
    assert.equal(hash, '#lambdaMode=scale&featureScale=0.15');
    assert.deepEqual((await hashToState(hash, DEFAULTS)).state, st);
    for (const bad of ['0', '1.5', 'x'])
        assert.deepEqual((await hashToState(`#featureScale=${bad}`, DEFAULTS)).rejected, ['featureScale']);
});
