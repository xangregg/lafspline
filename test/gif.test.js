import { test } from 'node:test';
import assert from 'node:assert/strict';
import { encodeGif, gifPalette, indexPixels } from '../src/core/gif.js';
import { createRng } from '../src/core/rng.js';

/**
 * A minimal GIF decoder for these tests, written from the GIF89a specification independently of the encoder: the
 * global palette, the frames' delays and LZW-decoded indices, and the loop count.
 */
function decodeGif(bytes) {
    let p = 0;
    const u8 = () => bytes[p++];
    const u16 = () => {
        const v = bytes[p] | (bytes[p + 1] << 8);
        p += 2;
        return v;
    };
    const text = n => String.fromCharCode(...bytes.subarray(p, (p += n)));
    assert.equal(text(6), 'GIF89a');
    const width = u16();
    const height = u16();
    const packed = u8();
    p += 2;
    const tableSize = 1 << ((packed & 7) + 1);
    const palette = bytes.slice(p, p + 3 * tableSize);
    p += 3 * tableSize;
    const frames = [];
    let delay = 0;
    let loop = null;
    for (;;) {
        const block = u8();
        if (block === 0x3b)
            break;
        if (block === 0x21) {
            const label = u8();
            if (label === 0xf9) {
                p++;
                p++;
                delay = u16() / 100;
                p += 2;
            }
            else if (label === 0xff) {
                const size = u8();
                const name = text(size);
                const sub = u8();
                if (name === 'NETSCAPE2.0') {
                    p++;
                    loop = u16();
                    p += sub - 3;
                }
                else {
                    p += sub;
                }
                assert.equal(u8(), 0);
            }
            continue;
        }
        assert.equal(block, 0x2c);
        p += 8;
        assert.equal(u8(), 0, 'no local table');
        const minCode = u8();
        const data = [];
        for (let n = u8(); n > 0; n = u8()) {
            data.push(...bytes.subarray(p, p + n));
            p += n;
        }
        frames.push({ delay, indices: lzwDecode(data, minCode, width * height) });
    }
    return { width, height, palette, frames, loop };
}

function lzwDecode(data, minCode, count) {
    const clear = 1 << minCode;
    const eoi = clear + 1;
    let size = minCode + 1;
    let dict = [];
    const reset = () => {
        dict = [];
        for (let i = 0; i < clear; i++)
            dict.push([i]);
        dict.push(null, null);
        size = minCode + 1;
    };
    reset();
    let bit = 0;
    const read = () => {
        let v = 0;
        for (let i = 0; i < size; i++, bit++)
            v |= ((data[bit >> 3] >> (bit & 7)) & 1) << i;
        return v;
    };
    const out = [];
    let prev = null;
    while (bit + size <= 8 * data.length) {
        const code = read();
        if (code === clear) {
            reset();
            prev = null;
            continue;
        }
        if (code === eoi)
            break;
        let entry;
        if (prev === null) {
            entry = dict[code];
        }
        else {
            entry = code < dict.length ? dict[code] : [...dict[prev], dict[prev][0]];
            if (dict.length < 4096)
                dict.push([...dict[prev], entry[0]]);
        }
        out.push(...entry);
        prev = code;
        if (dict.length === 1 << size && size < 12)
            size++;
    }
    assert.equal(out.length, count, 'decoded pixel count');
    return Uint8Array.from(out);
}

function rgbaOf(colors, pixels) {
    const out = new Uint8ClampedArray(4 * pixels.length);
    pixels.forEach((c, i) => {
        out.set([...colors[c], 255], 4 * i);
    });
    return out;
}

test('a GIF round-trips: palette, indices, delays and looping (decoded independently)', () => {
    const width = 37;
    const height = 23;
    const colors = [[252, 252, 251], [42, 120, 214], [235, 104, 52], [137, 135, 129], [11, 11, 11]];
    const rng = createRng(5);
    const frames = [0, 1, 2].map(() => rgbaOf(colors, Array.from({ length: width * height },
        () => Math.floor(rng.uniform() * colors.length))));
    const palette = gifPalette(frames);
    assert.equal(palette.length, 3 * colors.length, 'exact: every color kept');
    const cache = new Map();
    const indexed = frames.map(f => indexPixels(f, palette, cache));
    const delays = [3.5, 3.5, 6];
    const gif = encodeGif({ width, height, palette, frames: indexed.map((indices, k) => ({ indices,
        delay: delays[k] })) });
    const back = decodeGif(gif);
    assert.equal(back.width, width);
    assert.equal(back.height, height);
    assert.equal(back.loop, 0, 'loops forever');
    assert.deepEqual(back.frames.map(f => f.delay), delays);
    back.frames.forEach((f, k) => {
        // decoded colors equal the frame's pixels exactly
        for (let i = 0; i < width * height; i++) {
            const j = f.indices[i];
            assert.deepEqual([...back.palette.subarray(3 * j, 3 * j + 3)], [...frames[k].subarray(4 * i, 4 * i + 3)]);
        }
    });
});

test('a long frame passes the 4096-code table limit and its clear codes, and decodes exactly', () => {
    // random indices over 256 colors fill the table every few thousand pixels
    const width = 300;
    const height = 200;
    const rng = createRng(11);
    const indices = Uint8Array.from({ length: width * height }, () => Math.floor(rng.uniform() * 256));
    const palette = Uint8Array.from({ length: 3 * 256 }, (_, i) => (i * 7) & 255);
    const gif = encodeGif({ width, height, palette, frames: [{ indices, delay: 1 }] });
    assert.deepEqual(decodeGif(gif).frames[0].indices, indices);
    // and a flat frame, whose codes grow long fast
    const flat = new Uint8Array(width * height).fill(3);
    const gif2 = encodeGif({ width, height, palette, frames: [{ indices: flat, delay: 1 }] });
    assert.deepEqual(decodeGif(gif2).frames[0].indices, flat);
    assert.ok(gif2.length < 2000, `a flat frame compresses (${gif2.length} bytes)`);
});

test('two colors use the minimum code size; one pixel encodes', () => {
    const palette = Uint8Array.from([0, 0, 0, 255, 255, 255]);
    const indices = Uint8Array.from([0, 1, 1, 0, 1, 0]);
    assert.deepEqual(decodeGif(encodeGif({ width: 3, height: 2, palette, frames: [{ indices, delay: 0 }] }))
        .frames[0].indices, indices);
    const one = encodeGif({ width: 1, height: 1, palette, frames: [{ indices: Uint8Array.of(1), delay: 0 }] });
    assert.deepEqual([...decodeGif(one).frames[0].indices], [1]);
});

test('loop: null leaves out the looping extension, so the GIF plays once', () => {
    const palette = Uint8Array.from([0, 0, 0, 255, 255, 255]);
    const frames = [{ indices: Uint8Array.of(0, 1), delay: 1 }, { indices: Uint8Array.of(1, 0), delay: 2 }];
    const once = decodeGif(encodeGif({ width: 2, height: 1, palette, frames, loop: null }));
    assert.equal(once.loop, null);
    assert.equal(once.frames.length, 2);
    assert.equal(decodeGif(encodeGif({ width: 2, height: 1, palette, frames })).loop, 0, 'default: forever');
});

test('more than 256 colors: median cut keeps a few-pixel line color distinct from the background', () => {
    // a gradient background (thousands of colors) with a thin pure-blue line: the line's pixels must map to a color
    // within 8 per channel of it (the histogram's 5-bit cells are 8 wide), not to a background gray
    const width = 256;
    const height = 64;
    const rgba = new Uint8ClampedArray(4 * width * height);
    for (let yy = 0; yy < height; yy++) {
        for (let xx = 0; xx < width; xx++) {
            const p = 4 * (yy * width + xx);
            const line = yy === 30;
            rgba.set(line ? [42, 120, 214, 255] : [xx, (xx + yy) & 255, 255 - xx, 255], p);
        }
    }
    const palette = gifPalette([rgba], 256);
    assert.ok(palette.length <= 3 * 256);
    const idx = indexPixels(rgba, palette);
    const j = idx[30 * width + 5];
    const got = [...palette.subarray(3 * j, 3 * j + 3)];
    assert.ok(Math.max(...got.map((v, ch) => Math.abs(v - [42, 120, 214][ch]))) <= 8, `line color ${got}`);
});

test('invalid sizes and palettes are rejected', () => {
    const palette = Uint8Array.from([0, 0, 0]);
    assert.throws(() => encodeGif({ width: 0, height: 1, palette, frames: [] }), RangeError);
    assert.throws(() => encodeGif({ width: 1, height: 1, palette, frames: [] }), RangeError);
    assert.throws(() => encodeGif({ width: 2, height: 1, palette, frames: [{ indices: Uint8Array.of(0), delay: 0 }] }),
        RangeError);
    assert.throws(() => gifPalette([new Uint8Array(4)], 1), RangeError);
});
