import React from 'react';
import { ArrowRight } from 'lucide-react';

const blogPosts = [
  {
    image: "https://images.unsplash.com/photo-1542744173-8e7e53415bb0?auto=format&fit=crop&w=800&q=80",
    title: "Navigating International Trade Regulations",
    excerpt: "Learn about the latest updates in international trade policies and how they affect Indian exporters.",
    date: "March 15, 2024"
  },
  {
    image: "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?auto=format&fit=crop&w=800&q=80",
    title: "Digital Transformation in Export Business",
    excerpt: "Discover how technology is revolutionizing the export industry and creating new opportunities.",
    date: "March 12, 2024"
  },
  {
    image: "https://images.unsplash.com/photo-1560472355-536de3962603?auto=format&fit=crop&w=800&q=80",
    title: "Sustainable Packaging Solutions",
    excerpt: "Explore eco-friendly packaging options that meet international standards and reduce environmental impact.",
    date: "March 10, 2024"
  }
];

export default function Blog() {
  return (
    <section className="py-16 sm:py-20">
      <div className="container mx-auto px-4 sm:px-6">
        <div className="text-center mb-12">
          <h2 className="text-2xl sm:text-3xl font-bold mb-4">Latest Insights</h2>
          <p className="text-gray-600">Stay updated with the latest trends and insights in international trade</p>
        </div>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {blogPosts.map((post, index) => (
            <div key={index} className="bg-white rounded-xl overflow-hidden shadow-lg hover:shadow-xl transition-shadow">
              <img src={post.image} alt={post.title} className="w-full h-48 object-cover" />
              <div className="p-6">
                <p className="text-sm text-gray-500 mb-2">{post.date}</p>
                <h3 className="text-xl font-bold mb-3">{post.title}</h3>
                <p className="text-gray-600 mb-4">{post.excerpt}</p>
                <button className="text-[#FF9900] font-semibold inline-flex items-center">
                  Read More <ArrowRight className="ml-1 w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}