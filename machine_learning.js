const { loadImage, createCanvas, Image } = require('canvas');
const fs = require("fs");
const { promisify } = require("util");
const { Facenet } = require("facenet");

const facenet = new Facenet();

const readdir = promisify(fs.readdir);
const writeFile = promisify(fs.writeFile);
const readFile = promisify(fs.readFile);

let time = Date.now();
function log(message) {
    console.log(message + ` in ${Date.now() - time}ms`);
    time = Date.now();
}

async function train(images, isBase64) {
    log('Started training');
    
    const vectors = [];
    for (const image of images) {
        // find the vector associated with the face in the image
        const vector = await imageToVector(image, isBase64);
        vectors.push(vector);
    }
    
    log('Converted images to vectors');
    
    // Find the average of all the vectors
    const averageVector = Array(vectors[0].length).fill(0);
    for (const vector of vectors) {
        averageVector = addVectors(averageVector, vector);
    }
    for (const [index, value] of averageVector.entries()) {
        averageVector[index] = value / vectors.length;
    }
    
    // save the model
    const jsonData = JSON.stringify({
        vector: averageVector,
        numImages: images.length,
    });
    const existingModels = await readdir('./models');
    const filename = getRandomFilename('.json', existingModels);
    await writeFile(`./models/${filename}`, jsonData);
    
    log('Saved model');
    
    return filename;
}

async function predict(model, image, isBase64) { // model is a filename pointing to the saved model
    // Load the model
    const data = JSON.parse(await readFile(`./models/${model}`));
    
    const vector = await imageToVector(image, isBase64);
    
    return distanceBetween(data.vector, vector);
}

async function imageToVector(image, isBase64) {
    // need to convert image string to type ImageData
    const actualImage = await createImage(image, isBase64);
    
    // find face from image (we assume that there is only one, and get the first face found)
    const face = await facenet.align(actualImage);
    
    // find the vector associated with the face
    const vector = await facenet.embedding(face);
    
    return vector
}

function createImage(src, isBase64) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => {
            const canvas = createCanvas(image.width, image.height);
            const ctx = canvas.getContext('2d');
            ctx.drawImage(image, 0, 0);
            resolve(ctx.getImageData(0, 0, image.width, image.height));
        }
        image.onerror = () => reject('Error loading image');
        image.src = (isBase64 ? 'data:image/png;base64,' : '') + src;
    });
}

function addVectors(v1, v2, multiplier = 1) {
    // we assume that they are the same size
    const v1Copy = v1.slice(0);
    for (const [index, value] of v2.entries()) {
        v1Copy[index] += multiplier * value;
    }
    return v1Copy;
}

const subtractVectors = (v1, v2) => addVectors(v1, v2, -1);

function distanceBetween(v1, v2) {
    const difference = subtractVectors(v1, v2);
    const sumOfSquares = difference
        .map(x => Math.pow(x, 2))
        .reduce((total, value) => total + value, 0);
    return Math.sqrt(sumOfSquares);
}

function getRandomFilename(extension = '', existingFilenames = []) {
    let name = getRandomString(10) + extension;
    while (existingFilenames.indexOf(name) > -1) { // check for conflicts
        name = getRandomString(10) + extension;
    }
    return name;
}

function getRandomString(length) {
    const randString = () => Math.random().toString(36).substring(2, 15);
    let str = '';
    while (str.length < length) {
        str += randString();
    }
    return str.substr(0, length);
}

module.exports = { train, predict };