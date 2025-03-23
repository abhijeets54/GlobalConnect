import React from 'react';
import Hero from './components/Hero';
import Features from './components/Features';
import Blog from './components/Blog';
import Reviews from './components/Reviews';
import Footer from './components/Footer';

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      <Hero />
      <Features />
      <Blog />
      <Reviews />
      <Footer />
    </div>
  );
}

export default App;