const tf = require("@tensorflow/tfjs-node");
const knnClassifier = require("@tensorflow-models/knn-classifier");
const mobilenet = require("@tensorflow-models/mobilenet");
const { loadImage, createCanvas, Image } = require('canvas');
const fs = require("fs");
const { promisify } = require("util");

const readdir = promisify(fs.readdir);
const writeFile = promisify(fs.writeFile);
const readFile = promisify(fs.readFile);

async function train(images, isBase64) {
    
}

async function predict(model, image, isBase64) { // model is a filename pointing to the saved model

}

function createImage(src, isBase64) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => {
            const canvas = createCanvas(image.width, image.height);
            const ctx = canvas.getContext('2d');
            ctx.drawImage(image, 0, 0);
            resolve(canvas);
        }
        image.onerror = () => reject('Error loading image');
        image.src = (isBase64 ? 'data:image/png;base64,' : '') + src;
    });
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

// Modified from https://github.com/tensorflow/tfjs/issues/633#issuecomment-456308218
async function save(classifier, location) {
   let dataset = classifier.getClassifierDataset()
   var datasetObj = {}
   
   Object.keys(dataset).forEach((key) => {
     let data = dataset[key].dataSync();
     // use Array.from() so when JSON.stringify() it covert to an array string e.g [0.1,-0.2...] 
     // instead of object e.g {0:"0.1", 1:"-0.2"...}
     datasetObj[key] = Array.from(data);
   });
   
   let jsonStr = JSON.stringify(datasetObj);
   await writeFile(location, jsonStr);
 }
 
// Modified from https://github.com/tensorflow/tfjs/issues/633#issuecomment-456308218
async function load(location) {
    const classifier = knnClassifier.create();
    const dataset = await readFile(location);
    
    let tensorObj = JSON.parse(dataset)
    //covert back to tensor
    Object.keys(tensorObj).forEach((key) => {
      tensorObj[key] = tf.tensor(tensorObj[key], [tensorObj[key].length / 1024, 1024])
    });
    
    classifier.setClassifierDataset(tensorObj);
    return classifier;
}


module.exports = { train, predict };