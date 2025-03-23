import React from 'react';
import { Star } from 'lucide-react';

const reviews = [
  {
    name: "Rajesh Kumar",
    company: "Spice Exports Co.",
    image: "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?auto=format&fit=crop&w=150&q=80",
    content: "This platform has transformed how we manage our exports. The automated documentation and real-time tracking features have saved us countless hours.",
    rating: 5
  },
  {
    name: "Priya Sharma",
    company: "Textile Traders Ltd.",
    image: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?auto=format&fit=crop&w=150&q=80",
    content: "The analytics dashboard provides invaluable insights into our export operations. We've seen a 40% increase in efficiency since joining.",
    rating: 5
  },
  {
    name: "Anand Patel",
    company: "Organic India Exports",
    image: "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?auto=format&fit=crop&w=150&q=80",
    content: "Outstanding support team and intuitive interface. It's made international shipping and documentation management seamless.",
    rating: 5
  }
];

export default function Reviews() {
  return (
    <section className="bg-gray-50 py-16 sm:py-20">
      <div className="container mx-auto px-4 sm:px-6">
        <h2 className="text-2xl sm:text-3xl font-bold text-center mb-12">What Our Users Say</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {reviews.map((review, index) => (
            <div key={index} className="bg-white rounded-xl p-6 shadow-lg">
              <div className="flex items-center mb-4">
                <img src={review.image} alt={review.name} className="w-12 h-12 rounded-full mr-4" />
                <div>
                  <h3 className="font-bold">{review.name}</h3>
                  <p className="text-sm text-gray-600">{review.company}</p>
                </div>
              </div>
              <div className="flex mb-4">
                {[...Array(review.rating)].map((_, i) => (
                  <Star key={i} className="w-4 h-4 fill-current text-[#FF9900]" />
                ))}
              </div>
              <p className="text-gray-600">{review.content}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}