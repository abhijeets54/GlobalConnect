import React from 'react';
import { Globe, TrendingUp, FileText, MessageCircle, Shield, Users } from 'lucide-react';

const features = [
  {
    icon: <Globe className="w-8 h-8 text-[#FF9900]" />,
    title: "Global Reach",
    description: "Connect with verified shipping carriers worldwide and expand your business across borders"
  },
  {
    icon: <TrendingUp className="w-8 h-8 text-[#FF9900]" />,
    title: "Smart Analytics",
    description: "Make data-driven decisions with real-time insights and performance metrics"
  },
  {
    icon: <FileText className="w-8 h-8 text-[#FF9900]" />,
    title: "Document Management",
    description: "Streamline export documentation with automated compliance checks"
  },
  {
    icon: <MessageCircle className="w-8 h-8 text-[#FF9900]" />,
    title: "Real-time Communication",
    description: "Stay connected with carriers and customers through integrated messaging"
  },
  {
    icon: <Shield className="w-8 h-8 text-[#FF9900]" />,
    title: "Secure Platform",
    description: "Enterprise-grade security to protect your business data and transactions"
  },
  {
    icon: <Users className="w-8 h-8 text-[#FF9900]" />,
    title: "Community Support",
    description: "Join a thriving community of exporters and share best practices"
  }
];

export default function Features() {
  return (
    <section className="py-16 sm:py-20">
      <div className="container mx-auto px-4 sm:px-6">
        <h2 className="text-2xl sm:text-3xl font-bold text-center mb-12">Why Choose Our Platform?</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 sm:gap-12">
          {features.map((feature, index) => (
            <div key={index} className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow">
              <div className="mb-4">{feature.icon}</div>
              <h3 className="text-xl font-bold mb-3">{feature.title}</h3>
              <p className="text-gray-600">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}