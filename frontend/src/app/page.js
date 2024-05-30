'use client'

import { useState, useRef } from 'react';

const ImageUpload = () => {
  const [message, setMessage] = useState('');
  const [webcamMessage, setWebcamMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [webcamLoading, setWebcamLoading] = useState(false);
  const videoRef = useRef(null);
  const [cameraActive, setCameraActive] = useState(false);
  const canvasRef = useRef(null);

  const startWebcam = () => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
          setCameraActive(true);
        }
      })
      .catch(error => {
        console.error('Error', error);
        setMessage('Error');
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
    setWebcamLoading(true);
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      setWebcamMessage(result.message || 'Image uploaded successfully');
      setWebcamLoading(false);
    } catch (error) {
      console.error('Error', error);
      setWebcamMessage('Error');
      setWebcamLoading(false);
    }
  };

  const uploadImage = async (event) => {
    event.preventDefault();

    const input = document.getElementById('imageInput');
    if (!input.files || input.files.length === 0) {
      alert("Izaberite sliku prvo!");
      return;
    }

    const formData = new FormData();
    formData.append('image', input.files[0]);

    setLoading(true);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      setMessage(result.message || 'Image uploaded successfully');
      setLoading(false);
    } catch (error) {
      console.error('Error', error);
      setMessage('Error');
      setLoading(false);
    }
  };

  return (
    <div className='w-full flex flex-col-reverse justify-evenly items-center gap-y-2 min-h-screen h-full py-4 px-4'>
      <div className='flex flex-col justify-start items-center gap-y-4 border-dashed border-neutral-500 rounded-2xl border-2 p-4 w-full max-w-sm'>
        <h2 className='text-center font-semibold text-xl'>Otpremi sliku</h2>
        <input type="file" id="imageInput" className="hidden" />
        <label htmlFor="imageInput" className="text-center w-full custom-file-upload bg-neutral-500 text-white rounded-lg px-3 py-2">Otpremi</label>
        <button onClick={uploadImage} className='rounded-lg bg-blue-700 w-full text-white px-4 py-2'>Testiraj</button>
        {(message || loading) && <p>Rezultat: {loading ? 'Učitavanje...' : message}</p>}
      </div>
      <p className='py-0'>ili</p>
      <div className='flex flex-col justify-end items-center gap-y-4 border-dashed border-neutral-500 rounded-2xl border-2 p-4 w-full max-w-sm'>
        <h2 className='text-center font-semibold text-xl'>Koristi web kameru</h2>
        <video playsInline muted autoPlay controlsList="nodownload nofullscreen noremoteplayback" ref={videoRef} className="w-full rounded-lg" />
        {!cameraActive && <button className='bg-neutral-500 rounded-lg py-2 px-4 w-full text-white' onClick={startWebcam}>Uključi kameru</button>}
        {cameraActive && <button className='bg-blue-700 rounded-lg py-2 px-4 w-full text-white' onClick={captureImage}>Slikaj</button>}
        <canvas ref={canvasRef} className='hidden'></canvas>
        {(webcamMessage || webcamLoading) && <p>Rezultat: {webcamLoading ? 'Učitavanje...' : webcamMessage}</p>}
      </div>
    </div>
  );
};

export default ImageUpload;
