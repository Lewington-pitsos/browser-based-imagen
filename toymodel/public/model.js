TOKEN_LENGTH = 256

function getEncoderInput(inputIds) {
    const inputIdsTensor = new ort.Tensor("int64", new BigInt64Array(inputIds.map(x => BigInt(x))), [1, inputIds.length]);
    const encoderAttentionMaskTensor = new ort.Tensor("int64", new BigInt64Array(inputIds.length).fill(1n), [1, inputIds.length]);
    
    return [inputIdsTensor, encoderAttentionMaskTensor];
}

async function collectData() {
    const data = await (await fetch('tokenizer.json')).json()
    tokenizer = Tokenizer.fromConfig(data);
    console.log(data);
  
    const modelBuffer = await (await fetch('t5-model.onnx')).arrayBuffer()
    const sessionPromise = await ort.InferenceSession.create(modelBuffer, { executionProviders: ["wasm"] });
    transformerSession = await sessionPromise;
  
    console.log(transformerSession);
  
  
    let inputIds = tokenizer.encode("In the year 2525 if man is still alive, if woman can survive they will find");

    if (inputIds.length > TOKEN_LENGTH) {
        throw new Error(`expected input to be less than 256 tokens long, got ${inputIds}`);

    }

    const [inputIdsTensor, encoderAttentionMaskTensor] = getEncoderInput(inputIds);
    const encoderFeeds = {
        "tokens": inputIdsTensor,
        "attention_mask": encoderAttentionMaskTensor,
    }
    const encoderResults = await transformerSession.run(encoderFeeds);
    console.log("encoding:", encoderResults);  
}

console.log("about to collect data")
collectData();