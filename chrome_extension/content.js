// RAG Chrome Plugin - Content Script
// Runs on every webpage to extract content and highlight text

console.log('[RAG Plugin] Content script loaded');

// ==================== Content Extraction ====================

function extractPageContent() {
  /**
   * Extract clean text content from the current page
   * Similar to perceptionCP.py's extract_text_from_html
   */
  
  // Remove unwanted elements
  const unwantedSelectors = [
    'script', 'style', 'nav', 'footer', 'header', 
    'iframe', 'noscript', '.ad', '.advertisement'
  ];
  
  // Clone body to avoid modifying the actual page
  const bodyClone = document.body.cloneNode(true);
  
  // Remove unwanted elements
  unwantedSelectors.forEach(selector => {
    bodyClone.querySelectorAll(selector).forEach(el => el.remove());
  });
  
  // Get text content
  let text = bodyClone.innerText || bodyClone.textContent || '';
  
  // Clean up whitespace
  text = text
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.length > 0)
    .join('\n');
  
  return text;
}

function getPageTitle() {
  return document.title || 'Untitled';
}

function getPageUrl() {
  return window.location.href;
}

// ==================== Text Highlighting ====================

let currentHighlights = [];

function highlightText(textToHighlight) {
  /**
   * Highlight text on the current page using multiple strategies
   */
  
  console.log('[RAG Plugin] Attempting to highlight text:', textToHighlight.substring(0, 100) + '...');
  
  // Remove previous highlights
  removeHighlights();
  
  if (!textToHighlight || textToHighlight.length < 10) {
    console.warn('[RAG Plugin] Text too short to highlight reliably');
    return false;
  }
  
  try {
    // Get words from text (filter out very short ones)
    const words = textToHighlight.split(/\s+/).filter(w => w.length > 2);
    
    // Create multiple search patterns with different strategies
    const searchPatterns = [];
    
    // Strategy 1: Try progressively shorter patterns from the START
    for (let len of [15, 10, 7, 5, 4, 3]) {
      if (words.length >= len) {
        searchPatterns.push(words.slice(0, len).join(' '));
      }
    }
    
    // Strategy 2: Try patterns from the MIDDLE (in case start is generic)
    const midStart = Math.floor(words.length / 3);
    for (let len of [10, 7, 5]) {
      if (words.length >= midStart + len) {
        searchPatterns.push(words.slice(midStart, midStart + len).join(' '));
      }
    }
    
    // Strategy 3: Try the END (in case text is truncated at start)
    for (let len of [10, 7, 5]) {
      if (words.length >= len) {
        searchPatterns.push(words.slice(-len).join(' '));
      }
    }
    
    console.log(`[RAG Plugin] Will try ${searchPatterns.length} search patterns`);
    
    // Get all page text once for quick checking
    const bodyText = document.body.innerText || document.body.textContent || '';
    const normalizedBodyText = bodyText.replace(/\s+/g, ' ').toLowerCase();
    
    for (let i = 0; i < searchPatterns.length; i++) {
      const pattern = searchPatterns[i];
      if (pattern.length < 6) continue;
      
      const normalizedPattern = pattern.replace(/\s+/g, ' ').toLowerCase().trim();
      
      console.log(`[RAG Plugin] Pattern ${i+1}/${searchPatterns.length}: "${normalizedPattern.substring(0, 40)}..."`);
      
      // Quick check: is this pattern on the page at all?
      if (!normalizedBodyText.includes(normalizedPattern)) {
        console.log('[RAG Plugin]   ✗ Pattern not found in page text');
        continue;
      }
      
      console.log('[RAG Plugin]   ✓ Pattern exists! Finding nodes...');
      
      // Find all text nodes
      const walker = document.createTreeWalker(
        document.body,
        NodeFilter.SHOW_TEXT,
        {
          acceptNode: function(node) {
            // Skip script, style, and hidden elements
            if (node.parentElement.tagName === 'SCRIPT' || 
                node.parentElement.tagName === 'STYLE' ||
                node.parentElement.tagName === 'NOSCRIPT') {
              return NodeFilter.FILTER_REJECT;
            }
            
            // Skip if parent is not visible
            const parent = node.parentElement;
            if (parent) {
              const style = window.getComputedStyle(parent);
              if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
                return NodeFilter.FILTER_REJECT;
              }
            }
            
            return NodeFilter.FILTER_ACCEPT;
          }
        },
        false
      );
      
      let node;
      const allTextNodes = [];
      
      // Collect all visible text nodes
      while (node = walker.nextNode()) {
        const text = node.textContent.trim();
        if (text.length > 0) {
          allTextNodes.push(node);
        }
      }
      
      // Now search for pattern across text nodes
      for (let startIdx = 0; startIdx < allTextNodes.length; startIdx++) {
        let accumulatedText = '';
        const nodesToHighlight = [];
        
        // Try to build up text from consecutive nodes
        for (let endIdx = startIdx; endIdx < Math.min(startIdx + 15, allTextNodes.length); endIdx++) {
          const currentNode = allTextNodes[endIdx];
          accumulatedText += ' ' + currentNode.textContent;
          nodesToHighlight.push(currentNode);
          
          const normalizedAccumulated = accumulatedText.replace(/\s+/g, ' ').toLowerCase().trim();
          
          // Check if we found a match
          if (normalizedAccumulated.includes(normalizedPattern)) {
            console.log(`[RAG Plugin]   ✓✓ MATCH! Highlighting ${nodesToHighlight.length} nodes`);
            
            // Highlight these nodes
            nodesToHighlight.forEach(n => {
              if (n.parentNode && n.textContent.trim().length > 0) {
                try {
                  const span = document.createElement('span');
                  span.style.backgroundColor = '#00ff41';  // Bright neon green
                  span.style.color = '#000000';  // Black text for contrast
                  span.style.padding = '2px 4px';
                  span.style.borderRadius = '3px';
                  span.style.boxShadow = '0 0 10px rgba(0, 255, 65, 0.8)';
                  span.style.transition = 'all 0.3s ease';
                  span.style.fontWeight = '500';  // Slightly bold
                  span.className = 'rag-highlight';
                  span.textContent = n.textContent;
                  
                  n.parentNode.replaceChild(span, n);
                  currentHighlights.push(span);
                } catch (e) {
                  console.warn('[RAG Plugin] Could not replace node:', e);
                }
              }
            });
            
            // Scroll to first highlight with smooth animation
            if (currentHighlights.length > 0) {
              currentHighlights[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
              
              // Pulse effect - intensify the glow
              setTimeout(() => {
                currentHighlights.forEach(h => {
                  h.style.backgroundColor = '#00ff41';
                  h.style.boxShadow = '0 0 20px rgba(0, 255, 65, 1)';
                });
              }, 300);
              
              setTimeout(() => {
                currentHighlights.forEach(h => {
                  h.style.boxShadow = '0 0 12px rgba(0, 255, 65, 0.7)';
                });
              }, 600);
            }
            
            return true;
          }
        }
      }
    }
    
    // If no match found, show notification
    console.warn('[RAG Plugin] Could not find text to highlight on page');
    showNotification('Text found in index but not visible on current page. Try scrolling or checking different sections.');
    
    return false;
  } catch (error) {
    console.error('[RAG Plugin] Highlighting error:', error);
    return false;
  }
}

function showNotification(message) {
  /**
   * Show a temporary notification on the page
   */
  const notification = document.createElement('div');
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: #323232;
    color: white;
    padding: 16px 24px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    z-index: 10000;
    max-width: 300px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 14px;
    line-height: 1.4;
  `;
  notification.textContent = message;
  document.body.appendChild(notification);
  
  setTimeout(() => {
    notification.style.opacity = '0';
    notification.style.transition = 'opacity 0.3s';
    setTimeout(() => notification.remove(), 300);
  }, 4000);
}

function removeHighlights() {
  /**
   * Remove all current highlights
   */
  currentHighlights.forEach(span => {
    if (span.parentNode) {
      const textNode = document.createTextNode(span.textContent);
      span.parentNode.replaceChild(textNode, span);
    }
  });
  currentHighlights = [];
}

// ==================== Auto-Indexing ====================

let pageIndexed = false;
let autoIndexTimer = null;

async function checkAndAutoIndex() {
  /**
   * Check if page should be auto-indexed
   * This would be called after page loads
   */
  
  // Don't auto-index certain URLs
  const excludedPatterns = [
    'mail.google.com',
    'gmail.com',
    'web.whatsapp.com',
    'youtube.com',
    'chrome://',
    'chrome-extension://'
  ];
  
  const url = getPageUrl();
  
  // Check if URL should be indexed
  if (excludedPatterns.some(pattern => url.includes(pattern))) {
    console.log('[RAG Plugin] Page excluded from auto-indexing');
    return;
  }
  
  // For now, don't auto-index - let user trigger manually
  // In future, could add auto-index with notification
  console.log('[RAG Plugin] Page ready for indexing:', url);
}

// ==================== Message Listener ====================

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('[RAG Plugin] Message received:', request.action);
  
  if (request.action === 'getPageContent') {
    // Extract and return page content
    const content = extractPageContent();
    const title = getPageTitle();
    const url = getPageUrl();
    
    sendResponse({
      content: content,
      title: title,
      url: url,
      success: true
    });
    
    return true; // Keep channel open for async response
  }
  
  if (request.action === 'highlightText') {
    // Highlight text on page
    const success = highlightText(request.text);
    sendResponse({ success: success });
    return true;
  }
  
  if (request.action === 'removeHighlights') {
    // Remove highlights
    removeHighlights();
    sendResponse({ success: true });
    return true;
  }
  
  if (request.action === 'checkIfIndexed') {
    // Check if current page is indexed
    sendResponse({ indexed: pageIndexed });
    return true;
  }
});

// ==================== Page Load Handler ====================

// Check if we should auto-index when page loads
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    autoIndexTimer = setTimeout(checkAndAutoIndex, 2000);
  });
} else {
  autoIndexTimer = setTimeout(checkAndAutoIndex, 2000);
}

// Clean up on page unload
window.addEventListener('beforeunload', () => {
  if (autoIndexTimer) {
    clearTimeout(autoIndexTimer);
  }
  removeHighlights();
});

console.log('[RAG Plugin] Content script initialized');

