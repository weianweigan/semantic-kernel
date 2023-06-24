using Microsoft.SemanticKernel.Memory;
using System;

namespace Microsoft.SemanticKernel.Connectors.Memory.Milvus;

/// <summary>
/// Milvus memory store interface
/// </summary>
public interface IMilvusMemoryStore : IMemoryStore, IDisposable
{
}
