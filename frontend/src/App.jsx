import React, { useState } from 'react';
import SpecForm from './components/SpecForm';
import LayoutViewer from './components/LayoutViewer';
import SecurityReport from './components/SecurityReport';

function App() {
  const [pcbData, setPcbData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('generate');

  const handleGenerate = async (spec) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/v1/generate_layout', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ specification: spec }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        setPcbData(result.pcb_graph);
        setActiveTab('preview');
      } else {
        setError(result.message || 'Failed to generate layout');
      }
    } catch (error) {
      console.error('Error generating layout:', error);
      setError('Failed to generate layout: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeSecurity = async () => {
    if (!pcbData) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/v1/analyze_security', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ pcb_graph: pcbData }),
      });

      const result = await response.json();
      
      if (result.overall_score !== undefined) {
        // Update PCB data with security info
        setPcbData(prev => ({
          ...prev,
          security_analysis: result
        }));
      }
    } catch (error) {
      setError('Security analysis failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    if (!pcbData) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/v1/export_kicad', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pcb_graph: pcbData,
          filename: `pcb_layout_${Date.now()}.kicad_pcb`
        }),
      });

      const result = await response.json();
      
      if (result.success) {
        alert(`Successfully exported to: ${result.filepath}`);
      } else {
        setError(result.message || 'Export failed');
      }
    } catch (error) {
      setError('Export failed: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header>
        <div className="header-content">
          <h1>
            <span style={{marginRight: '0.75rem'}}>⚡</span>
            PunRek - AI-Native EDA Platform
          </h1>
          <p>AI-powered PCB design and optimization</p>
        </div>
      </header>

      <main className="container">
        <div className="tabs">
          <button
            onClick={() => setActiveTab('generate')}
            className={`tab-button ${activeTab === 'generate' ? 'active' : ''}`}
          >
            Generate Layout
          </button>
          <button
            onClick={() => setActiveTab('preview')}
            disabled={!pcbData}
            className={`tab-button ${activeTab === 'preview' && pcbData ? 'active' : ''} ${!pcbData ? 'disabled' : ''}`}
          >
            Preview Layout
          </button>
        </div>

        {error && (
          <div className="error-message">
            🚨 {error}
          </div>
        )}

        {activeTab === 'generate' && (
          <div className="grid grid-cols-2">
            <div className="card">
              <h2 className="card-title">PCB Specifications</h2>
              <SpecForm onGenerate={handleGenerate} loading={loading} />
            </div>
            
            <div className="card">
              <h2 className="card-title">Quick Start</h2>
              <div className="space-y-3">
                <div style={{padding: '1rem', backgroundColor: '#dbeafe', borderRadius: '0.5rem'}}>
                  <h3 style={{fontWeight: '500', color: '#1e40af'}}>Sample Configuration</h3>
                  <p style={{fontSize: '0.875rem', color: '#3730a3'}}>8 components, 2 layers, digital signals</p>
                </div>
                <button
                  onClick={() => handleGenerate({
                    component_count: 8,
                    max_trace_length: 25.0,
                    layers: 2,
                    power_domains: ['3.3V'],
                    signal_types: ['digital'],
                    constraints: { min_clearance: 0.2 }
                  })}
                  disabled={loading}
                  className="btn btn-primary"
                  style={{width: '100%'}}
                >
                  {loading ? 'Generating...' : 'Generate Sample Layout'}
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'preview' && pcbData && (
          <div style={{display: 'flex', flexDirection: 'column', gap: '1.5rem'}}>
            <div className="card">
              <div className="flex justify-between items-center" style={{marginBottom: '1rem'}}>
                <h2 className="card-title">PCB Layout Preview</h2>
                <div className="btn-group">
                  <button
                    onClick={handleAnalyzeSecurity}
                    disabled={loading}
                    className="btn btn-secondary"
                    style={{padding: '0.25rem 0.75rem', fontSize: '0.875rem'}}
                  >
                    {loading ? 'Analyzing...' : 'Security Analysis'}
                  </button>
                  <button
                    onClick={handleExport}
                    disabled={loading}
                    className="btn btn-primary"
                    style={{padding: '0.25rem 0.75rem', fontSize: '0.875rem'}}
                  >
                    Export KiCad
                  </button>
                </div>
              </div>
              
              <LayoutViewer pcbData={pcbData} loading={loading} />
              
              {pcbData.security_analysis && (
                <SecurityReport report={pcbData.security_analysis} />
              )}
            </div>
          </div>
        )}

        {!pcbData && activeTab === 'preview' && (
          <div className="card center-content">
            <div style={{fontSize: '2rem', color: '#94a3b8', marginBottom: '1rem'}}>📊</div>
            <h3 style={{fontSize: '1.5rem', fontWeight: '600', color: '#64748b', marginBottom: '0.5rem'}}>No Layout Generated Yet</h3>
            <p style={{color: '#94a3b8', marginBottom: '1rem'}}>Generate a PCB layout first using the specifications form.</p>
            <button
              onClick={() => setActiveTab('generate')}
              className="btn btn-primary"
            >
              Go to Generate Tab
            </button>
          </div>
        )}
      </main>

      <footer>
        <div className="footer-content">
          <p>PunRek - AI-Native EDA Platform v1.0 | Backend running on http://localhost:8000</p>
        </div>
      </footer>
    </div>
  );
}

export default App;