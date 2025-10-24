const http = require('http');

const server = http.createServer((req, res) => {
  res.write('Hola desde Node.js puro');
  res.end();
});

server.listen(3000, () => {
  console.log('Servidor en http://localhost:3000');
});
