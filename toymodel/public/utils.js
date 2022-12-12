function getArticle(str) {
    // Check if the first letter of the string is a vowel
    if (/^[aeiou]/i.test(str)) {
      // If it is, return "an"
      return "an";
    } else {
      // If it isn't, return "a"
      return "a";
    }
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
  

function getEncoderInput(inputIds) {
    const inputIdsTensor = new ort.Tensor("int64", new BigInt64Array(inputIds.map(x => BigInt(x))), [1, inputIds.length]);
    const encoderAttentionMaskTensor = new ort.Tensor("int64", new BigInt64Array(inputIds.length).fill(1n), [1, inputIds.length]);
    
    return [inputIdsTensor, encoderAttentionMaskTensor];
}

async function updateIteration(i) {
    document.getElementById("count").textContent=(i + 1).toString();
    await sleep(10)
}