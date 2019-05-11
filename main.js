const tf = require("@tensorflow/tfjs-node");
const mobilenet = require("@tensorflow-models/mobilenet");
const canvas = require("canvas");
const express = require("express");

const app = express();
const port = 8080;
const host = '0.0.0.0';

app.get('/', (req, res) => {
    res.send('Hello World!');
})

app.listen(port, host, () => {
   console.log(`Listening on ${host}:${port}`); 
});