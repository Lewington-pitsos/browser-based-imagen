'use strict';

function loadModel() {
  console.log("content script in da house");
}

async function startContentScript() {
  console.log('going to execute script')
  
  const query = { active: true, lastFocusedWindow: true };
  let [tab] = await chrome.tabs.query(query);
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: loadModel,
  });


  console.log('executed script')
}

function goToPage() {
  chrome.tabs.create({ url: chrome.runtime.getURL("model.html") });
}

chrome.runtime.onInstalled.addListener(() => {
    // startContentScript();
    // collectData();
    // goToPage();
    console.log("background script loaded")
});


chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'GREETINGS') {
    const message = `Hi ${
      sender.tab ? 'Con' : 'Pop'
    }, DIE DIE DIE.`;

    // Log message coming from the `request` parameter
    console.log("request picked up by service worker", request.payload.message);
    // Send a response message
    sendResponse({
      message,
    });
  }
});
