const express = require('express');
const path = require('path');
const app = express();

// Servir frontend (si decides integrarlo en Render en el mismo proyecto)
app.use(express.static(path.join(__dirname, '../frontend')));

// Ruta simple
app.get('/api', (req, res) => {
  res.json({ mensaje: "Hola desde el backend" });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Servidor en puerto ${PORT}`));

