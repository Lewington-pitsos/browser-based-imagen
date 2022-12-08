TOKEN_LENGTH = 256

function getEncoderInput(inputIds) {
    // TODO work out how to actually calculate the encoder attention mask
    // What we have now is a stub 
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
  
    const unetBuffer = await (await fetch('unet-32.onnx')).arrayBuffer()
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

    const batch_size = 1;
    const channels = 3;
    const width = 32;
    const height = 32;

    const noise = new ort.Tensor(
        'float32', 
        Array.from({length: batch_size * channels * width * height}, () => Math.floor(Math.random())), 
        [batch_size, channels, width, height]
    );
    
    const unetFeeds = {
        'image': noise, 
        'text_embeds': encoderResults.encoding,
        'text_mask': new ort.Tensor("bool", new  Uint8Array(inputIds.length).fill(true), [batch_size, inputIds.length]),
        'timestep': new ort.Tensor("float32", [0.4], [1]),
        'time_next': new ort.Tensor("float32", [0.404], [1]),
        // 'cond_scale': new ort.Tensor("float32", [1.], [1]),
    }

    const image = await unet.run(unetFeeds);
    console.log("encoding:", image);   
}


console.log("about to collect data")
collectData();