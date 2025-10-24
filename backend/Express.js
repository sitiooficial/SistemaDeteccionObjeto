const express = require('express');
const app = express();

app.get('/', (req, res) => {
  res.send('Hola desde Express.js');
});

app.listen(3000, () => {
  console.log('Servidor en http://localhost:3000');
});

