import React from 'react';
import { ArrowRight } from 'lucide-react';

export default function Hero() {
  return (
    <header className="bg-gradient-to-r from-[#232F3E] to-[#37475A] text-white">
      <div className="container mx-auto px-4 sm:px-6 py-16 sm:py-24">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold mb-6">Empowering Indian SMBs to Export Globally</h1>
          <p className="text-lg sm:text-xl mb-8">Streamline your international trade operations with our comprehensive export management platform</p>
          <button className="bg-[#FF9900] hover:bg-[#FF9900]/90 text-white font-bold py-3 px-8 rounded-full inline-flex items-center">
            Explore Platform <ArrowRight className="ml-2" />
          </button>
        </div>
      </div>
    </header>
  );
}