const { train, predict } = require("./machine_learning.js");
const fs = require("fs");
const { promisify } = require("util");

const readdir = promisify(fs.readdir);

async function main() {
    console.log('Testing');
    // const files = await readdir('./test_faces/training');
    // const filepaths = files.map(filename => './test_faces/training/' + filename);
    
    // const modelName = await train(filepaths.slice(0, 1), false);
    
    // console.log(modelName);
    
    // const testImage = './test_faces/test/'
    
    // const matchResults = await predict('sq7zpqyo2i.json', './test_faces/test/match.jpg', false);
    // const matchResults2 = await predict('sq7zpqyo2i.json', './test_faces/test/match2.jpg', false);
    // const failResults = await predict('sq7zpqyo2i.json', './test_faces/test/fail.jpg', false);
    
    // console.log(matchResults);
    // console.log(matchResults2);
    // console.log(failResults);
    
    console.log(await predict('gue72bdi7k.json', './test_faces/test/fail.jpg', false));
    return true;
}

main();
