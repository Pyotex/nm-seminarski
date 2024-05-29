'use client'

import { useState, useRef } from 'react';

const ImageUpload = () => {
  const [message, setMessage] = useState('');
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const startWebcam = () => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
      })
      .catch(error => {
        console.error('Error accessing webcam:', error);
        setMessage('Error accessing webcam');
      });
  };

  const captureImage = () => {
    if (videoRef.current && videoRef.current.readyState === 4 && canvasRef.current) {
      const canvas = canvasRef.current;
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      canvas.getContext('2d').drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('image', blob, 'webcam.jpg');

        uploadCapturedImage(formData);
      }, 'image/jpeg');
    }
  };

  const uploadCapturedImage = async (formData) => {
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      setMessage(result.message || 'Image uploaded successfully');
    } catch (error) {
      console.error('Error uploading image:', error);
      setMessage('Error uploading image');
    }
  };

  const uploadImage = async (event) => {
    event.preventDefault();

    const input = document.getElementById('imageInput');
    if (!input.files || input.files.length === 0) {
      alert("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append('image', input.files[0]);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      setMessage(result.message || 'Image uploaded successfully');
    } catch (error) {
      console.error('Error uploading image:', error);
      setMessage('Error uploading image');
    }
  };

  return (
    <div className='w-full flex flex-col justify-center items-center gap-y-2 h-screen'>
      <input type="file" id="imageInput" className="hidden" />
      <label htmlFor="imageInput" className="custom-file-upload bg-white text-black border border-green-600 rounded px-3 py-2 cursor-pointer transition duration-300 ease-in-out hover:bg-green-500">Ubaci sliku</label>
      <button onClick={uploadImage} className='rounded-md bg-white text-black px-4 py-2'>Testiraj</button>
      <br />
      <video ref={videoRef} className="w-[320px] h-[240px] rounded-lg" />
      <button onClick={startWebcam}>Uključi kameru</button>
      <button onClick={captureImage}>Slikaj</button>
      <canvas ref={canvasRef} className='hidden'></canvas>
      {message && <p>Rezultat: {message}</p>}
    </div>
  );
};

export default ImageUpload;
