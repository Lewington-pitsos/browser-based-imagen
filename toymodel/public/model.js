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
    console.log('tokenizer', tokenizer)
  
    const tfmBuffer = await (await fetch('t5-model.onnx')).arrayBuffer()
    const tfmSessionPromise = await ort.InferenceSession.create(tfmBuffer, { executionProviders: ["wasm"] });
  
    const unetBuffer = await (await fetch('unet.onnx')).arrayBuffer()
    const unetSessionPromise = await ort.InferenceSession.create(unetBuffer, { executionProviders: ["wasm"] });
    
    const transformer = await tfmSessionPromise;
    console.log('transformer session', transformer);
    
    unet = await unetSessionPromise;
    console.log('unet session', unet);
  
    let inputIds = tokenizer.encode("a photo of a truck");

    if (inputIds.length > TOKEN_LENGTH) {
        throw new Error(`expected input to be less than 256 tokens long, got ${inputIds}`);

    }

    const [inputIdsTensor, encoderAttentionMaskTensor] = getEncoderInput(inputIds);
    const encoderFeeds = {
        "tokens": inputIdsTensor,
        "attention_mask": encoderAttentionMaskTensor,
    }
    const encoderResults = await transformer.run(encoderFeeds);
    console.log("encoding:", encoderResults);



}

console.log("about to collect data")
collectData();