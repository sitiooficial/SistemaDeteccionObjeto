async function llamarBackend() {
  const respuesta = await fetch('/api');
  const data = await respuesta.json();
  document.getElementById('respuesta').innerText = data.mensaje;
}
