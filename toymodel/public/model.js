TOKEN_LENGTH = 256

function getEncoderInput(inputIds) {
    // TODO work out how to actually calculate the encoder attention mask
    // What we have now is a stub 
    const inputIdsTensor = new ort.Tensor("int64", new BigInt64Array(inputIds.map(x => BigInt(x))), [1, inputIds.length]);
    const encoderAttentionMaskTensor = new ort.Tensor("int64", new BigInt64Array(inputIds.length).fill(1n), [1, inputIds.length]);
    
    return [inputIdsTensor, encoderAttentionMaskTensor];
}

async function updateIteration(i) {
    document.getElementById("count").textContent=(i + 1).toString();
    await sleep(10)
}

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
  
function clamp(number, min, max) {
    return Math.max(min, Math.min(number, max));
}

function clampAndDenormalize(number, min, max) {
    return (clamp(number, min, max) + 1) * 0.5;
}


function getColorIndicesForCoord(x, y, width) {
    const red = y * (width * 4) + x * 4;
    return [red, red + 1, red + 2, red + 3];
};
  

async function displayOutput(data) {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    // const rgbValues = Uint8Array.from(data.map(x => clampAndDenormalize(x, -1, 1) * 255))
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

    console.log('steps', steps.length)
    
    for (let i = 0; i < timesteps; i++) {
        await updateIteration(i);
        const timeBound = steps[i];
        console.log(timeBound)

        const unetFeeds = {
            'image': img, 
            'text_embeds': text_embeds,
            'text_mask': text_mask,
            'time_next': new ort.Tensor("float32", timeBound[0], [batch_size]),
            'timestep': new ort.Tensor("float32", timeBound[1], [batch_size]),
            'cond_scale': cond_scale
        }

        const pred = await unet.run(unetFeeds);
        console.log(pred)
        img = pred.prediction;

        await displayOutput(img.data);
    }

    img = img.data.map(x => clampAndDenormalize(x, -1., 1.));

    return img
}


function randNorm() {
    // Generate a random number using the JavaScript Math.random() function
    var x = Math.random();
  
    // Use the Box-Muller transform to convert the random number to a number
    // with a normal distribution
    var y = Math.sqrt(-2 * Math.log(x)) * Math.cos(2 * Math.PI * x);
  
    // Return the number
    return y;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function collectData() {
    const batch_size = 1;
    const channels = 3;
    const width = 32;
    const height = 32;
    
    const noise = new ort.Tensor(
        'float32', 
        Array.from({length: batch_size * channels * width * height}, () => randNorm()), 
        [batch_size, channels, width, height]
    );
    
    // const noise = new ort.Tensor("float32", (await (await fetch('truck.json')).json())['img'].flat().flat().flat(), [1, 3, 32, 32])
    // const noise = (await (await fetch('truck.json')).json())['img']

    await displayOutput(noise.data);
    
    const data = await (await fetch('tokenizer.json')).json()
    tokenizer = Tokenizer.fromConfig(data);
    console.log('tokenizer loaded')
  
    const tfmBuffer = await (await fetch('t5-model.onnx')).arrayBuffer()
    const tfmSessionPromise = await ort.InferenceSession.create(tfmBuffer, { executionProviders: ["wasm"] });
    
    const unetBuffer = await (await fetch('unet-32.onnx')).arrayBuffer()
    const unetSessionPromise = await ort.InferenceSession.create(unetBuffer, { executionProviders: ["wasm"]});
    
    const transformer = await tfmSessionPromise;
    console.log('transformer loaded, extended');
    document.getElementById("status").textContent="Loading diffusion model...";
    await sleep(10)


    unet = await unetSessionPromise;
    console.log('unet session loaded');
    document.getElementById("status").textContent="Standing by";
    await sleep(10)
  





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

    console.log("starting inference with encoder results:", encoderResults.encoding);

    document.getElementById("status").textContent="Performing Inference...";
    await sleep(10)
    const inputData = await (await fetch('inputs.json')).json()
    output = await sample(
        unet, 
        new ort.Tensor("float32", inputData.image.flat().flat().flat(), [batch_size, channels, width, height]), 
        new ort.Tensor("float32", inputData.text_embeds.flat().flat(), [batch_size, inputIds.length, 768]), 
        new ort.Tensor("bool", inputData.text_mask.flat(), [batch_size, inputIds.length]),
        new ort.Tensor("float32", inputData.cond_scale, [batch_size]),
        250,
    )

    document.getElementById("status").textContent="Finished!";
    await sleep(10)

    // console.log("making prediction")
    // const inputData = await (await fetch('inputs.json')).json()
    // console.log('cond scale', inputData.cond_scale)
    // const unetFeeds = {
    //     'image':  
    //     'text_embeds': new ort.Tensor("float32", inputData.text_embeds.flat().flat(), [batch_size, inputIds.length, 768]),
    //     'text_mask': new ort.Tensor("bool", inputData.text_mask.flat(), [batch_size, inputIds.length]),
    //     'timestep': new ort.Tensor("float32", inputData.timestep, [batch_size]),
    //     'time_next': new ort.Tensor("float32", inputData.time_next, [batch_size]),
    //     'cond_scale': new ort.Tensor("float32", inputData.cond_scale, [batch_size])
    // }

    // const pred = await unet.run(unetFeeds);

    // console.log("prediction", Array.from(pred.prediction.data).flat().flat().flat())
    // console.log("expected", inputData.expected.flat().flat().flat())
}

collectData();