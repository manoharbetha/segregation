// script.js — robust preview + backend call
document.addEventListener('DOMContentLoaded', () => {
  const API = '/api/segregate';

  const fileInput = document.getElementById('waste-image-input');
  const chooseBtn = document.getElementById('choose-btn');
  const img = document.getElementById('image-preview');
  const placeholder = document.getElementById('preview-placeholder');
  const classifyBtn = document.getElementById('classify-button');
  const spinner = document.getElementById('loading-spinner');
  const errorMsg = document.getElementById('error-message');
  const results = document.getElementById('results-section');

  const cards = {
    "Biodegradable": {
      pct: document.getElementById('pct-biodegradable'),
      bar: document.getElementById('bar-biodegradable'),
      label: document.getElementById('label-biodegradable')
    },
    "Non-Biodegradable": {
      pct: document.getElementById('pct-non-biodegradable'),
      bar: document.getElementById('bar-non-biodegradable'),
      label: document.getElementById('label-non-biodegradable')
    },
    "E-Waste": {
      pct: document.getElementById('pct-e-waste'),
      bar: document.getElementById('bar-e-waste'),
      label: document.getElementById('label-e-waste')
    }
  };

  let selectedFile = null;
  let lastObjectUrl = null;

  // safe guards
  if (!fileInput || !chooseBtn || !img) {
    console.error('Preview elements missing: check IDs in HTML.');
    return;
  }

  // open native file dialog
  chooseBtn.addEventListener('click', (e) => {
    e.preventDefault();
    fileInput.click();
  });

  // when user selects a file
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) {
      console.warn('No file selected.');
      return;
    }
    selectedFile = file;

    // free previous object URL if any
    if (lastObjectUrl) {
      URL.revokeObjectURL(lastObjectUrl);
      lastObjectUrl = null;
    }

    // Try createObjectURL first (fast)
    try {
      const url = URL.createObjectURL(file);
      lastObjectUrl = url;
      showPreviewFromUrl(url);
    } catch (err) {
      console.warn('createObjectURL failed, falling back to FileReader', err);
      // fallback to FileReader data URL
      const fr = new FileReader();
      fr.onload = () => showPreviewFromUrl(fr.result);
      fr.onerror = (fe) => {
        console.error('FileReader error', fe);
        showPlaceholder();
      };
      fr.readAsDataURL(file);
    }

    classifyBtn.disabled = false;
    resetCards();
    if (results) results.classList.add('hidden');
  });

  function showPreviewFromUrl(url) {
    // ensure placeholder hidden only after image successfully loads
    img.style.opacity = '0';
    img.style.display = 'block';
    img.src = url;

    // set up onload handler to fade in *after* image is decoded
    img.onload = () => {
      // revoke object URL if applicable (do it after onload)
      if (lastObjectUrl && img.src === lastObjectUrl) {
        // revoke later to avoid some browser race conditions
        setTimeout(() => { try { URL.revokeObjectURL(lastObjectUrl); } catch(e){} lastObjectUrl = null; }, 100);
      }
      placeholder.style.display = 'none';
      // force a reflow then fade
      requestAnimationFrame(() => { img.classList.add('img-fade'); img.style.opacity = '1'; });
      console.info('Preview loaded:', img.src);
    };

    img.onerror = (ev) => {
      console.error('Image failed to load', ev);
      showPlaceholder();
    };
  }

  function showPlaceholder() {
    img.src = '';
    img.style.display = 'none';
    placeholder.style.display = 'block';
  }

  function resetCards() {
    errorMsg && errorMsg.classList.add('hidden');
    for (const k in cards) {
      if (!cards[k]) continue;
      cards[k].pct.textContent = '--%';
      if (cards[k].bar) cards[k].bar.style.width = '0%';
      if (cards[k].label) cards[k].label.textContent = 'Confidence: --';
    }
  }

  function setLoading(on) {
    if (!spinner) return;
    spinner.classList.toggle('hidden', !on);
    if (classifyBtn) classifyBtn.disabled = on;
  }

  async function callApi(file) {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch(API, { method: 'POST', body: fd });
    if (!r.ok) {
      let body;
      try { body = await r.json(); } catch(_) { body = null; }
      throw new Error((body && body.message) ? body.message : `HTTP ${r.status}`);
    }
    return r.json();
  }

  async function handleClassify() {
    if (!selectedFile) return;
    setLoading(true);
    try {
      const data = await callApi(selectedFile);

      // support both class_probs (percentage) or single category+confidence
      if (Array.isArray(data.class_probs) && data.class_probs.length) {
        const map = {};
        data.class_probs.forEach(it => {
          const name = (it.class || it.label || '').trim();
          const pct = (it.percentage !== undefined) ? Number(it.percentage) : (it.probability !== undefined ? Number(it.probability) * 100 : 0);
          if (name) map[name.toLowerCase()] = pct;
        });
        // update cards safely
        updateCardSafe('Biodegradable', map['biodegradable'] ?? map['biodegradable ' ] ?? 0);
        updateCardSafe('Non-Biodegradable', map['non-biodegradable'] ?? map['non biodegradable'] ?? 0);
        updateCardSafe('E-Waste', map['e-waste'] ?? map['e waste'] ?? 0);

      } else if (data.category && data.confidence !== undefined) {
        // fallback single label
        resetCards();
        const cat = data.category;
        const conf = Number(data.confidence) * 100;
        let key = 'Non-Biodegradable';
        if (/bio/i.test(cat)) key = 'Biodegradable';
        else if (/e-? ?waste/i.test(cat)) key = 'E-Waste';
        updateCardSafe(key, conf);
      } else {
        throw new Error('Unexpected server response format.');
      }

      // show results if present
      if (results) {
        results.classList.remove('hidden');
        results.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    } catch (err) {
      console.error('Classification error', err);
      if (errorMsg) {
        errorMsg.textContent = `Error: ${err.message || 'Could not classify'}`;
        errorMsg.classList.remove('hidden');
      }
    } finally {
      setLoading(false);
    }
  }

  function updateCardSafe(key, pct) {
    const safe = Math.max(0, Math.min(100, Number(pct) || 0));
    const entry = cards[key];
    if (!entry) return;
    entry.pct.textContent = `${safe.toFixed(2)}%`;
    if (entry.bar) entry.bar.style.width = `${safe}%`;
    if (entry.label) entry.label.textContent = `Confidence: ${safe.toFixed(2)}%`;
  }

  // attach classify handler
  if (classifyBtn) classifyBtn.addEventListener('click', handleClassify);

  // initial reset
  resetCards();
});
