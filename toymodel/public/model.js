TOKEN_LENGTH = 256

function getTimesteps(startValue, stopValue, cardinality, batch_size) {
    cardinality = cardinality + 1
    var arr = [];
    var step = (stopValue - startValue) / (cardinality - 1);
    for (var i = 0; i < cardinality; i++) {
      arr.push(startValue + (step * i));
    }
  
    arr = arr.reverse();
  
    const timesteps = [];
  
    for (let i = 0; i < arr.length - 1; i++) {
      const after = arr[i];
      const prev = arr[i+1];
      const tsBatch = []
      tsBatch.push(Array(batch_size).fill(prev))
      tsBatch.push(Array(batch_size).fill(after))
      timesteps.push(tsBatch)
    }
  
    return timesteps;
}
  
async function displayOutput(data) {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(32, 32);
    const width = 32
    const height = 32

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const [redIndex, greenIndex, blueIndex, alphaIndex] = getColorIndicesForCoord(x, y, width);
            
            const arrayIdx = x + y * width;

            imgData.data[redIndex] = clampAndDenormalize(data[arrayIdx], -1, 1) * 255
            imgData.data[greenIndex] = clampAndDenormalize(data[arrayIdx + width * height], -1, 1) * 255
            imgData.data[blueIndex] = clampAndDenormalize(data[arrayIdx + (2*width * height)], -1, 1) * 255
            imgData.data[alphaIndex] = 255;
      
        }
      }
      
    ctx.putImageData(imgData, 0, 0);

    await sleep(10)
}

async function sample(unet, img, text_embeds, text_mask, cond_scale, timesteps) {
    batch_size = img.dims[0]
  
    const steps = getTimesteps(0, 1, timesteps, batch_size).slice(1)

    for (let i = 0; i < timesteps; i++) {
        await updateIteration(i);
        const timeBound = steps[i];

        const unetFeeds = {
            'image': img, 
            'text_embeds': text_embeds,
            'text_mask': text_mask,
            'time_next': new ort.Tensor("float32", timeBound[0], [batch_size]),
            'timestep': new ort.Tensor("float32", timeBound[1], [batch_size]),
            'cond_scale': cond_scale
        }

        const pred = await unet.run(unetFeeds);
        img = pred.prediction;

        await displayOutput(img.data);
    }

    img = img.data.map(x => clampAndDenormalize(x, -1., 1.));

    return img
}

async function loadModels() {
    const data = await (await fetch('tokenizer.json')).json()
    const tokenizer = Tokenizer.fromConfig(data);
    console.log('tokenizer loaded')
  
    const tfmBuffer = await (await fetch('t5-model.onnx')).arrayBuffer()
    const tfmSessionPromise = await ort.InferenceSession.create(tfmBuffer, { executionProviders: ["wasm"] });
    
    const unetBuffer = await (await fetch('unet-32.onnx')).arrayBuffer()
    const unetSessionPromise = await ort.InferenceSession.create(unetBuffer, { executionProviders: ["wasm"]});
    
    const transformer = await tfmSessionPromise;
    console.log('transformer loaded, extended');
    document.getElementById("status").textContent="Loading diffusion model...";
    await sleep(10)


    const unet = await unetSessionPromise;
    console.log('unet session loaded');
    document.getElementById("status").textContent="Standing by";
    await sleep(10)

    return [tokenizer, transformer, unet]
}

async function performInference(tokenizer, transformer, unet, prompt) {
    const batch_size = 1;
    const channels = 3;
    const width = 32;
    const height = 32;
    
    const noise = new ort.Tensor(
        'float32', 
        Array.from({length: batch_size * channels * width * height}, () => randNorm()), 
        [batch_size, channels, width, height]
    );
    
    await displayOutput(noise.data);


    let inputIds = tokenizer.encode(prompt);

    if (inputIds.length > TOKEN_LENGTH) {
        throw new Error(`expected input to be less than 256 tokens long, got ${inputIds}`);
    }

    const [inputIdsTensor, encoderAttentionMaskTensor] = getEncoderInput(inputIds);
    const encoderFeeds = {
        "tokens": inputIdsTensor,
        "attention_mask": encoderAttentionMaskTensor,
    }
    const encoderResults = await transformer.run(encoderFeeds);

    console.log("starting inference with encoder results:", encoderResults.encoding);

    document.getElementById("status").textContent="Performing Inference...";
    await sleep(10)
    output = await sample(
        unet, 
        noise, 
        encoderResults.encoding,
        new ort.Tensor("bool", new  Uint8Array(inputIds.length).fill(true), [batch_size, inputIds.length]),
        new ort.Tensor("float32", [1.1], [batch_size]),        
        250,
    )

    document.getElementById("status").textContent="Finished!";
    await sleep(10)
}


async function submitForm(tokenizer, transformer, unet) {
    const promptSelector = document.getElementById("prompt")
    const prompt = promptSelector.value
    promptSelector.disabled = true;
    document.getElementById("submit").disabled = true;

    await performInference(tokenizer, transformer, unet, "a photo of " + getArticle(prompt) + " " + prompt)
    document.getElementById("prompt").disabled = false;
    document.getElementById("submit").disabled = false;
}

async function main() {
    const [tokenizer, transformer, unet ] = await loadModels();
    const submitPartial = submitForm.bind(null, tokenizer, transformer, unet);
    const submitElement = document.getElementById("submit")
    submitElement.disabled = false;
    document.getElementById("prompt").disabled=false

    submitElement.addEventListener("click", submitPartial);
}

main();
