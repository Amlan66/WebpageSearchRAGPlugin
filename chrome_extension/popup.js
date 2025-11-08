// RAG Chrome Plugin - Popup Script

const API_BASE = 'http://localhost:8000';

// DOM Elements
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const resultsBox = document.getElementById('resultsBox');
const statsBox = document.getElementById('statsBox');
const indexCurrentBtn = document.getElementById('indexCurrentBtn');
const refreshStatsBtn = document.getElementById('refreshStatsBtn');
const indexProgress = document.getElementById('indexProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');

// Stats elements
const statPages = document.getElementById('statPages');
const statChunks = document.getElementById('statChunks');
const statCurrent = document.getElementById('statCurrent');

// ==================== API Functions ====================

async function searchContent(query) {
  try {
    const response = await fetch(`${API_BASE}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        top_k: 5
      })
    });
    
    if (!response.ok) throw new Error('Search failed');
    return await response.json();
  } catch (error) {
    console.error('Search error:', error);
    throw error;
  }
}

async function getStatus() {
  try {
    const response = await fetch(`${API_BASE}/status`);
    if (!response.ok) throw new Error('Failed to get status');
    return await response.json();
  } catch (error) {
    console.error('Status error:', error);
    throw error;
  }
}

async function checkCurrentUrl(url) {
  try {
    const response = await fetch(`${API_BASE}/check_url?url=${encodeURIComponent(url)}`);
    if (!response.ok) throw new Error('Failed to check URL');
    return await response.json();
  } catch (error) {
    console.error('Check URL error:', error);
    return null;
  }
}

async function indexPage(url, title, content) {
  try {
    const response = await fetch(`${API_BASE}/index`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: url,
        title: title,
        content: content,
        force_reindex: false
      })
    });
    
    if (!response.ok) throw new Error('Indexing failed');
    return await response.json();
  } catch (error) {
    console.error('Index error:', error);
    throw error;
  }
}

// ==================== UI Functions ====================

function showLoading(message = 'Loading...') {
  resultsBox.style.display = 'block';
  resultsBox.innerHTML = `
    <div class="loading">
      <div class="spinner"></div>
      <div>${message}</div>
    </div>
  `;
}

function showEmptyState(message = 'No results found') {
  resultsBox.style.display = 'block';
  resultsBox.innerHTML = `
    <div class="empty-state">
      <div class="empty-state-icon">📭</div>
      <div>${message}</div>
    </div>
  `;
}

function showError(message) {
  resultsBox.style.display = 'block';
  resultsBox.innerHTML = `
    <div class="empty-state">
      <div class="empty-state-icon">⚠️</div>
      <div>${message}</div>
      <div style="font-size: 12px; margin-top: 10px; opacity: 0.7;">
        Make sure the local agent is running on port 8000
      </div>
    </div>
  `;
}

function displayResults(results) {
  if (!results || results.length === 0) {
    showEmptyState('No results found. Try indexing more pages!');
    return;
  }

  resultsBox.style.display = 'block';
  resultsBox.innerHTML = results.map(result => `
    <div class="result-item" data-url="${result.url}" data-chunk="${escapeHtml(result.chunk_text)}">
      <div class="result-title">${escapeHtml(result.title)}</div>
      <div class="result-url">${escapeHtml(result.url)}</div>
      <div class="result-snippet">${escapeHtml(result.chunk_text.substring(0, 150))}...</div>
    </div>
  `).join('');

  // Add click handlers
  document.querySelectorAll('.result-item').forEach(item => {
    item.addEventListener('click', () => {
      const url = item.dataset.url;
      const chunk = item.dataset.chunk;
      navigateToResult(url, chunk);
    });
  });
}

function updateStats(status, currentUrlInfo) {
  statPages.textContent = status.total_urls || 0;
  statChunks.textContent = status.total_chunks || 0;
  
  if (currentUrlInfo && currentUrlInfo.indexed) {
    statCurrent.textContent = `✅ Indexed (${currentUrlInfo.chunk_count} chunks)`;
    indexCurrentBtn.textContent = '🔄 Re-index Current Page';
  } else {
    statCurrent.textContent = '❌ Not indexed';
    indexCurrentBtn.textContent = '📄 Index Current Page';
  }
}

function showProgress(percentage, message) {
  indexProgress.style.display = 'block';
  progressFill.style.width = `${percentage}%`;
  progressText.textContent = message;
}

function hideProgress() {
  indexProgress.style.display = 'none';
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ==================== Action Functions ====================

async function performSearch(autoNavigate = false) {
  const query = searchInput.value.trim();
  
  if (!query) {
    showError('Please enter a search query');
    return;
  }

  searchBtn.disabled = true;
  showLoading('Searching...');

  try {
    const response = await searchContent(query);
    
    if (autoNavigate && response.results && response.results.length > 0) {
      // Automatically navigate to first result when Enter is pressed
      const topResult = response.results[0];
      navigateToResult(topResult.url, topResult.chunk_text);
    } else {
      // Just display results when button is clicked
      displayResults(response.results);
    }
  } catch (error) {
    showError('Search failed. Is the agent running?');
  } finally {
    searchBtn.disabled = false;
  }
}

async function indexCurrentPage() {
  indexCurrentBtn.disabled = true;
  showProgress(10, 'Getting page content...');

  try {
    // Get current tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (!tab.url.startsWith('http')) {
      showError('Cannot index this page (not a web page)');
      hideProgress();
      indexCurrentBtn.disabled = false;
      return;
    }

    showProgress(30, 'Extracting content...');

    // Get page content from content script
    const response = await chrome.tabs.sendMessage(tab.id, { action: 'getPageContent' });
    
    if (!response || !response.content) {
      throw new Error('Could not extract page content');
    }

    showProgress(50, 'Sending to agent...');

    // Send to agent for indexing
    const result = await indexPage(tab.url, response.title, response.content);

    showProgress(100, 'Complete!');
    
    // Show success message
    resultsBox.style.display = 'block';
    resultsBox.innerHTML = `
      <div style="text-align: center; padding: 20px; color: #4caf50;">
        <div style="font-size: 48px; margin-bottom: 10px;">✅</div>
        <div style="font-weight: 600; margin-bottom: 5px;">Page Indexed Successfully!</div>
        <div style="font-size: 12px; opacity: 0.8;">${result.chunks_created} chunks created</div>
      </div>
    `;

    // Refresh stats
    await refreshStats();

    setTimeout(hideProgress, 2000);
  } catch (error) {
    console.error('Indexing error:', error);
    showError(`Indexing failed: ${error.message}`);
    hideProgress();
  } finally {
    indexCurrentBtn.disabled = false;
  }
}

async function refreshStats() {
  try {
    const status = await getStatus();
    
    // Get current tab URL
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    let currentUrlInfo = null;
    
    if (tab && tab.url.startsWith('http')) {
      currentUrlInfo = await checkCurrentUrl(tab.url);
    }
    
    updateStats(status, currentUrlInfo);
  } catch (error) {
    console.error('Stats refresh error:', error);
  }
}

async function navigateToResult(url, chunkText) {
  try {
    // Send navigation request to background script
    chrome.runtime.sendMessage({
      action: 'navigateAndHighlight',
      url: url,
      text: chunkText
    });
    
    // Close popup
    window.close();
  } catch (error) {
    console.error('Navigation error:', error);
  }
}

// ==================== Event Listeners ====================

// Button click - show results list
searchBtn.addEventListener('click', () => performSearch(false));

// Enter key - auto-navigate to top result
searchInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') {
    performSearch(true);  // Auto-navigate on Enter
  }
});

indexCurrentBtn.addEventListener('click', indexCurrentPage);

refreshStatsBtn.addEventListener('click', refreshStats);

// ==================== Initialize ====================

async function initialize() {
  // Load initial stats
  await refreshStats();
  
  // Focus search input
  searchInput.focus();
}

// Run on load
initialize();

