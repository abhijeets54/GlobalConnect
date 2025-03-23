import React from 'react';

export default function Footer() {
  return (
    <footer className="bg-[#232F3E] text-white py-12">
      <div className="container mx-auto px-4 sm:px-6">
        <div className="grid md:grid-cols-3 gap-8 mb-8">
          <div>
            <h3 className="font-bold mb-4">About Us</h3>
            <p className="text-gray-300 text-sm">Empowering Indian SMBs with cutting-edge export management solutions to compete globally.</p>
          </div>
          <div>
            <h3 className="font-bold mb-4">Quick Links</h3>
            <ul className="text-gray-300 text-sm space-y-2">
              <li><a href="#" className="hover:text-[#FF9900]">Features</a></li>
              <li><a href="#" className="hover:text-[#FF9900]">Success Stories</a></li>
              <li><a href="#" className="hover:text-[#FF9900]">Blog</a></li>
              <li><a href="#" className="hover:text-[#FF9900]">Contact</a></li>
            </ul>
          </div>
          <div>
            <h3 className="font-bold mb-4">Contact</h3>
            <ul className="text-gray-300 text-sm space-y-2">
              <li>Email: support@amazonsmb.in</li>
              <li>Phone: +91 1800 XXX XXXX</li>
              <li>Location: Bangalore, India</li>
            </ul>
          </div>
        </div>
        <div className="border-t border-gray-700 pt-8 text-center">
          <p className="text-sm text-gray-300">© 2024 Amazon India SMB Export Platform. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
}