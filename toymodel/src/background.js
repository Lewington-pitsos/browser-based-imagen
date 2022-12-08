'use strict';

import * as ort from 'onnxruntime-web';
import { Tokenizer } from './tokenizers.js';

// With background scripts you can communicate with popup
// and contentScript files.
// For more information on background script,
// See https://developer.chrome.com/extensions/background_pages



async function collectData() {
  console.log(chrome.runtime.getURL('tokenizer.json'))
  const data = await (await fetch(chrome.runtime.getURL('tokenizer.json'))).json()
  console.log(data);

  const modelBuffer = await (await fetch(chrome.runtime.getURL('t5-model.onnx'))).arrayBuffer()
  const sessionPromise = await ort.InferenceSession.create(modelBuffer, { executionProviders: ["wasm"] });
  transformerSession = await sessionPromise;

  console.log(transformerSession);

  // let inputIds = tokenizer.encode("In the year 2525 if man is still alive, if woman can survive they will find");

  // if (inputIds.length > TOKEN_LENGTH) {
  //     throw new Error(`expected input to be less than 256 tokens long, got ${inputIds}`);
  // }

  // inputIdsTensor, encoderAttentionMaskTensor = getEncoderInput(inputIds);

  // const encoderFeeds = {
  //     "tokens": inputIdsTensor,
  //     "attention_mask": encoderAttentionMaskTensor,
  // }
  // const encoderResults = await transformer.run(encoderFeeds);
  // console.log("encoding:", encoderResults.data);

}

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
