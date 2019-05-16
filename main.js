const express = require("express");
const { train, predict } = require('./machine_learning.js');

const app = express();
const port = 3000;
// const host = '0.0.0.0';
// const host = '127.0.0.1';

app.use(express.json({limit: '16mb', strict: false}));

app.get('/', (req, res) => {
    res.send('Hello World!');
});

app.post('/train', (req, res) => {
    console.log(req.headers['content-length'] / 1000 + 'kB')
    const images = req.body.images;
    train(images, true).then(modelName => res.send(modelName));
});

app.post('/predict', (req, res) => {
    console.log(req.headers['content-length'] / 1000 + 'kB');
    const {model, image} = req.body;
    predict(model, image, true).then(distance => res.send(String(distance)));
});

app.listen(port, () => {
   console.log(`Listening on port ${port}`); 
});
