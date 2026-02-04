import React, { useState } from 'react';

const SpecForm = ({ onGenerate, loading }) => {
  const [spec, setSpec] = useState({
    component_count: 8,
    max_trace_length: 25.0,
    layers: 2,
    power_domains: ['3.3V'],
    signal_types: ['digital'],
    constraints: { min_clearance: 0.2 }
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    onGenerate(spec);
  };

  const updateField = (field, value) => {
    setSpec(prev => ({ ...prev, [field]: value }));
  };

  const updateArrayItem = (arrayName, index, value) => {
    setSpec(prev => ({
      ...prev,
      [arrayName]: prev[arrayName].map((item, i) => i === index ? value : item)
    }));
  };

  const addArrayItem = (arrayName) => {
    setSpec(prev => ({
      ...prev,
      [arrayName]: [...prev[arrayName], '']
    }));
  };

  const removeArrayItem = (arrayName, index) => {
    setSpec(prev => ({
      ...prev,
      [arrayName]: prev[arrayName].filter((_, i) => i !== index)
    }));
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="bg-gray-50 p-4 rounded-lg">
        <h3 className="text-lg font-medium text-gray-800 mb-4">Basic Specifications</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Component Count</label>
            <input
              type="number"
              value={spec.component_count}
              onChange={(e) => updateField('component_count', parseInt(e.target.value) || 0)}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              min="1"
              max="100"
              required
            />
            <p className="mt-1 text-xs text-gray-500">Number of components (1-100)</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Max Trace Length (mm)</label>
            <input
              type="number"
              step="0.1"
              value={spec.max_trace_length}
              onChange={(e) => updateField('max_trace_length', parseFloat(e.target.value) || 0)}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              min="1"
              max="100"
              required
            />
            <p className="mt-1 text-xs text-gray-500">Maximum trace length allowed</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Layers</label>
            <select
              value={spec.layers}
              onChange={(e) => updateField('layers', parseInt(e.target.value))}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
            >
              <option value={1}>1 Layer</option>
              <option value={2}>2 Layers</option>
              <option value={4}>4 Layers</option>
              <option value={6}>6 Layers</option>
            </select>
            <p className="mt-1 text-xs text-gray-500">PCB layer count</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Min Clearance (mm)</label>
            <input
              type="number"
              step="0.01"
              value={spec.constraints.min_clearance}
              onChange={(e) => updateField('constraints', { ...spec.constraints, min_clearance: parseFloat(e.target.value) || 0.1 })}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              min="0.01"
              max="1"
            />
            <p className="mt-1 text-xs text-gray-500">Minimum spacing between traces</p>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 p-4 rounded-lg">
        <h3 className="text-lg font-medium text-blue-800 mb-4">Power Domains</h3>
        <div className="space-y-3">
          {spec.power_domains.map((domain, index) => (
            <div key={index} className="flex items-center space-x-2">
              <input
                type="text"
                value={domain}
                onChange={(e) => updateArrayItem('power_domains', index, e.target.value)}
                placeholder={`Power domain ${index + 1} (e.g., 3.3V)`}
                className="flex-1 p-3 border border-blue-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
              {spec.power_domains.length > 1 && (
                <button
                  type="button"
                  onClick={() => removeArrayItem('power_domains', index)}
                  className="p-2 text-red-600 hover:text-red-800 hover:bg-red-100 rounded-full"
                >
                  ×
                </button>
              )}
            </div>
          ))}
          <button
            type="button"
            onClick={() => addArrayItem('power_domains')}
            className="w-full p-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
          >
            + Add Power Domain
          </button>
        </div>
      </div>

      <div className="bg-green-50 p-4 rounded-lg">
        <h3 className="text-lg font-medium text-green-800 mb-4">Signal Types</h3>
        <div className="space-y-3">
          {spec.signal_types.map((type, index) => (
            <div key={index} className="flex items-center space-x-2">
              <input
                type="text"
                value={type}
                onChange={(e) => updateArrayItem('signal_types', index, e.target.value)}
                placeholder={`Signal type ${index + 1} (e.g., digital)`}
                className="flex-1 p-3 border border-green-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
              />
              {spec.signal_types.length > 1 && (
                <button
                  type="button"
                  onClick={() => removeArrayItem('signal_types', index)}
                  className="p-2 text-red-600 hover:text-red-800 hover:bg-red-100 rounded-full"
                >
                  ×
                </button>
              )}
            </div>
          ))}
          <button
            type="button"
            onClick={() => addArrayItem('signal_types')}
            className="w-full p-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm font-medium"
          >
            + Add Signal Type
          </button>
        </div>
      </div>

      <div className="flex space-x-4 pt-4">
        <button
          type="submit"
          disabled={loading}
          className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-700 text-white py-4 px-6 rounded-lg hover:from-blue-700 hover:to-indigo-800 disabled:from-gray-400 disabled:cursor-not-allowed font-medium text-lg transition-all transform hover:scale-[1.02]"
        >
          {loading ? (
            <div className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 4.955 4.045 9 9 9v-4.001z"></path>
              </svg>
              Generating PCB Layout...
            </div>
          ) : 'Generate PCB Layout'}
        </button>
        
        <button
          type="button"
          onClick={() => setSpec({
            component_count: 8,
            max_trace_length: 25.0,
            layers: 2,
            power_domains: ['3.3V'],
            signal_types: ['digital'],
            constraints: { min_clearance: 0.2 }
          })}
          className="px-6 py-4 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors font-medium"
        >
          Reset
        </button>
      </div>

      <div className="bg-gray-50 p-4 rounded-lg">
        <h4 className="font-medium text-gray-700 mb-2">Quick Presets</h4>
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={() => setSpec({
              component_count: 6,
              max_trace_length: 20.0,
              layers: 2,
              power_domains: ['3.3V'],
              signal_types: ['digital'],
              constraints: { min_clearance: 0.2 }
            })}
            className="text-sm p-2 bg-blue-100 text-blue-800 rounded hover:bg-blue-200"
          >
            Small Digital Board
          </button>
          <button
            type="button"
            onClick={() => setSpec({
              component_count: 12,
              max_trace_length: 35.0,
              layers: 4,
              power_domains: ['3.3V', '5V', '1.8V'],
              signal_types: ['digital', 'analog', 'clock'],
              constraints: { min_clearance: 0.15 }
            })}
            className="text-sm p-2 bg-purple-100 text-purple-800 rounded hover:bg-purple-200"
          >
            Medium Mixed-Signal
          </button>
        </div>
      </div>
    </form>
  );
};

export default SpecForm;