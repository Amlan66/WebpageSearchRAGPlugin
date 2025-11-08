// RAG Chrome Plugin - Background Service Worker
// Handles navigation, coordination between content scripts and popup

console.log('[RAG Plugin] Background service worker started');

const API_BASE = 'http://localhost:8000';

// ==================== State Management ====================

let agentStatus = {
  connected: false,
  lastCheck: null
};

// ==================== Agent Communication ====================

async function checkAgentConnection() {
  /**
   * Check if local agent is running
   */
  try {
    const response = await fetch(`${API_BASE}/`);
    if (response.ok) {
      agentStatus.connected = true;
      agentStatus.lastCheck = Date.now();
      console.log('[RAG Plugin] Agent connected');
      return true;
    }
  } catch (error) {
    agentStatus.connected = false;
    console.warn('[RAG Plugin] Agent not reachable');
  }
  return false;
}

async function notifyAgentOfPageVisit(url, title) {
  /**
   * Notify agent when user visits a page
   * Agent can decide whether to index
   */
  try {
    const response = await fetch(`${API_BASE}/check_url?url=${encodeURIComponent(url)}`);
    if (response.ok) {
      const data = await response.json();
      return data;
    }
  } catch (error) {
    console.error('[RAG Plugin] Failed to notify agent:', error);
  }
  return null;
}

// ==================== Navigation & Highlighting ====================

async function navigateAndHighlight(url, text) {
  /**
   * Navigate to a URL and highlight text
   */
  try {
    console.log('[RAG Plugin] Navigating to:', url);
    console.log('[RAG Plugin] Text to highlight:', text.substring(0, 100) + '...');
    
    // First, check if tab with URL already exists
    const tabs = await chrome.tabs.query({ url: url });
    
    let targetTab;
    let tabExisted = false;
    
    if (tabs.length > 0) {
      // Tab exists, switch to it
      console.log('[RAG Plugin] Tab already exists, switching to it');
      targetTab = tabs[0];
      tabExisted = true;
      await chrome.tabs.update(targetTab.id, { active: true });
      await chrome.windows.update(targetTab.windowId, { focused: true });
      
      // For existing tabs, wait a bit for tab to be fully ready
      await new Promise(resolve => setTimeout(resolve, 500));
    } else {
      // Create new tab
      console.log('[RAG Plugin] Creating new tab');
      targetTab = await chrome.tabs.create({ url: url, active: true });
      
      // Wait for page to load completely
      await new Promise(resolve => {
        const listener = (tabId, changeInfo) => {
          if (tabId === targetTab.id && changeInfo.status === 'complete') {
            chrome.tabs.onUpdated.removeListener(listener);
            console.log('[RAG Plugin] Page loaded completely');
            resolve();
          }
        };
        chrome.tabs.onUpdated.addListener(listener);
        
        // Timeout after 10 seconds
        setTimeout(() => {
          chrome.tabs.onUpdated.removeListener(listener);
          console.log('[RAG Plugin] Page load timeout, proceeding anyway');
          resolve();
        }, 10000);
      });
      
      // Extra wait for content script to initialize
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    // Send highlight command to content script with retry
    let retries = 3;
    let success = false;
    
    while (retries > 0 && !success) {
      try {
        console.log(`[RAG Plugin] Attempting to highlight (${4 - retries}/3)...`);
        
        const response = await chrome.tabs.sendMessage(targetTab.id, {
          action: 'highlightText',
          text: text
        });
        
        if (response && response.success) {
          console.log('[RAG Plugin] Highlight successful');
          success = true;
        } else {
          console.warn('[RAG Plugin] Highlight failed, retrying...');
          await new Promise(resolve => setTimeout(resolve, 500));
          retries--;
        }
      } catch (error) {
        console.warn('[RAG Plugin] Message send failed:', error.message);
        await new Promise(resolve => setTimeout(resolve, 500));
        retries--;
      }
    }
    
    if (!success) {
      console.error('[RAG Plugin] Could not highlight text after 3 attempts');
      // Show notification to user
      try {
        chrome.notifications.create({
          type: 'basic',
          title: 'RAG Plugin',
          message: 'Navigated to page but could not highlight text. Try reloading the page.',
          priority: 1
        });
      } catch (err) {
        // Notifications permission might not be granted
        console.log('[RAG Plugin] Could not show notification:', err.message);
      }
    }
    
    console.log('[RAG Plugin] Navigation complete');
    return success;
  } catch (error) {
    console.error('[RAG Plugin] Navigation error:', error);
    return false;
  }
}

// ==================== Auto-Indexing Logic ====================

let visitedPages = new Set();

async function handlePageVisit(tabId, url, title) {
  /**
   * Handle when user visits a page
   * Decide whether to auto-index
   */
  
  // Skip if already visited in this session
  if (visitedPages.has(url)) {
    return;
  }
  
  visitedPages.add(url);
  
  // Check with agent
  const urlInfo = await notifyAgentOfPageVisit(url, title);
  
  if (urlInfo && !urlInfo.indexed) {
    // Page not indexed - could show notification
    // For now, just log
    console.log('[RAG Plugin] New page available for indexing:', url);
    
    // Could show badge on extension icon
    try {
      await chrome.action.setBadgeText({ text: '!', tabId: tabId });
      await chrome.action.setBadgeBackgroundColor({ color: '#4CAF50', tabId: tabId });
    } catch (error) {
      // Ignore badge errors
    }
  }
}

// ==================== Message Handlers ====================

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('[RAG Plugin] Background received message:', request.action);
  
  if (request.action === 'navigateAndHighlight') {
    // Navigate to URL and highlight text
    navigateAndHighlight(request.url, request.text)
      .then(success => sendResponse({ success: success }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true; // Keep channel open for async
  }
  
  if (request.action === 'checkAgent') {
    // Check if agent is connected
    checkAgentConnection()
      .then(connected => sendResponse({ connected: connected }))
      .catch(() => sendResponse({ connected: false }));
    return true;
  }
  
  if (request.action === 'getAgentStatus') {
    // Return agent status
    sendResponse(agentStatus);
    return true;
  }
});

// ==================== Tab Event Listeners ====================

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  // When page finishes loading
  if (changeInfo.status === 'complete' && tab.url) {
    // Only handle http/https URLs
    if (tab.url.startsWith('http')) {
      handlePageVisit(tabId, tab.url, tab.title);
    }
  }
});

chrome.tabs.onActivated.addListener(async (activeInfo) => {
  // When user switches tabs, clear badge
  try {
    await chrome.action.setBadgeText({ text: '', tabId: activeInfo.tabId });
  } catch (error) {
    // Ignore
  }
});

// ==================== Extension Install/Update ====================

chrome.runtime.onInstalled.addListener((details) => {
  console.log('[RAG Plugin] Extension installed/updated:', details.reason);
  
  if (details.reason === 'install') {
    // First install - could open welcome page
    console.log('[RAG Plugin] First install - welcome!');
    
    // Check agent connection
    checkAgentConnection();
  } else if (details.reason === 'update') {
    // Extension updated
    console.log('[RAG Plugin] Extension updated');
  }
});

// ==================== Periodic Agent Check ====================

// Check agent connection every 30 seconds
setInterval(() => {
  checkAgentConnection();
}, 30000);

// Initial check
checkAgentConnection();

console.log('[RAG Plugin] Background service worker initialized');

