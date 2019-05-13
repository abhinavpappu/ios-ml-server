const express = require("express");
const { train, predict } = require('./machine_learning.js');

const app = express();
const port = 8080;
// const host = '0.0.0.0';
const host = '127.0.0.1';

app.get('/', (req, res) => {
    res.send('Hello World!');
})

app.post('/train', (req, res) => {
    const images = JSON.parse(req.query.images);
    train(images, true).then(modelName => res.send(modelName));
});

app.post('/predict', (req, res) => {
    const image = req.query.image;
    predict(image, true).then(distance => res.send(distance));
})

app.listen(port, host, () => {
   console.log(`Listening on ${host}:${port}`); 
});