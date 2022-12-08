const ort = require('onnxruntime-node');

async function main() {
    try {
        console.log("Loading model...");
        const session = await ort.InferenceSession.create('./t5-model.onnx');
        
        console.log("Model loaded.", session);
        
        // for (let index = 8; index < 11; index++) {   
        //     const tensorA = new ort.Tensor('int64', BigInt64Array.from([BigInt(index)]), [1]);
        //     const feeds = {a: tensorA};
        //     const result = await session.run(feeds);
        //     console.log(result.b.data[0])
        // }

    } catch (e) {
        console.error(`failed to inference ONNX model: ${e}.`);
    }
}

main();