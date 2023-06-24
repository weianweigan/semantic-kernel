using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Grpc.Net.Client;
using IO.Milvus;
using IO.Milvus.Client;
using IO.Milvus.Client.gRPC;
using IO.Milvus.Client.REST;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.SemanticKernel.AI.Embeddings;
using Microsoft.SemanticKernel.Memory;

namespace Microsoft.SemanticKernel.Connectors.Memory.Milvus;

/// <summary>
/// An implementation of <see cref="IMemoryStore"/> for Milvus Vector database.
/// </summary>
/// <remarks>
/// The Embedding data is saved to a Milvus Vector database instance that the client is connected to.
/// </remarks>
public class MilvusMemoryStore : IMilvusMemoryStore
{
    private const string EmbeddingFieldName = "embedding";
    private const string IdFieldName = "id";
    private const string MetadataFieldName = "metadata";
    private readonly IMilvusClient _milvusClient;
    private readonly int _vectorSize;
    private readonly MilvusIndexType _milvusIndexType;
    private readonly ILogger _log;
    private bool _disposedValue;

    ///<inheritdoc/>
    public IMilvusClient MilvusClient => this._milvusClient;

    /// <summary>
    /// Construct a milvus memory store.
    /// </summary>
    /// <param name="milvusClient">Milvus client.</param>
    /// <param name="vectorSize">Vector size.</param>
    /// <param name="milvusIndexType">Milvus index type. AutoIndex for zilliz cloud or milvus 2.2.9 above</param>
    /// <param name="log">Instance of logger.</param>
    public MilvusMemoryStore(
        IMilvusClient milvusClient,
        int vectorSize,
        MilvusIndexType milvusIndexType = MilvusIndexType.AUTOINDEX,
        ILogger? log = null)
    {
        this._log = log ?? NullLogger<MilvusMemoryStore>.Instance;
        this._milvusClient = milvusClient;
        this._vectorSize = vectorSize;
        this._milvusIndexType = milvusIndexType;
    }

    /// <summary>
    /// Construct a milvus memory store.
    /// </summary>
    /// <param name="host">Milvus server address.</param>
    /// <param name="port">Port</param>
    /// <param name="vectorSize">Vector size</param>
    /// <param name="milvusIndexType">Milvus index type. AutoIndex for zilliz cloud or milvus 2.2.9 above</param>
    /// <param name="log">Instance of logger.</param>
    public MilvusMemoryStore(
        string host,
        int port,
        int vectorSize,
        MilvusIndexType milvusIndexType = MilvusIndexType.AUTOINDEX,
        ILogger? log = null)
    {
        this._vectorSize = vectorSize;
        this._milvusIndexType = milvusIndexType;
        this._log = log ?? NullLogger<MilvusMemoryStore>.Instance;
        this._milvusClient = new MilvusGrpcClient(host, port, log: log);
    }

    /// <summary>
    /// Construct a milvus memory store.
    /// </summary>
    /// <param name="host">Milvus memory store.</param>
    /// <param name="port">Port.</param>
    /// <param name="vectorSize">Vector size.</param>
    /// <param name="userName">Username.</param>
    /// <param name="password">Password.</param>
    /// <param name="milvusIndexType">Milvus index type. AutoIndex for zilliz cloud or milvus 2.2.9 above</param>
    /// <param name="grpcChannel">Grpc channel</param>
    /// <param name="log">Logger.</param>
    public MilvusMemoryStore(
        string host,
        int port,
        int vectorSize,
        string userName = "root",
        string password = "milvus",
        MilvusIndexType milvusIndexType = MilvusIndexType.AUTOINDEX,
        GrpcChannel? grpcChannel = null,
        ILogger? log = null)
    {
        this._vectorSize = vectorSize;
        this._milvusIndexType = milvusIndexType;
        this._log = log ?? NullLogger<MilvusMemoryStore>.Instance;
        this._milvusClient = new MilvusGrpcClient(host, port, userName, password, log: log, grpcChannel: grpcChannel);
    }

    /// <summary>
    /// Construct a milvus memory store.
    /// </summary>
    /// <param name="host">Milvus memory store.</param>
    /// <param name="port">Port.</param>
    /// <param name="vectorSize">Vector size.</param>
    /// <param name="userName">Username.</param>
    /// <param name="password">Password.</param>
    /// <param name="milvusIndexType"></param>
    /// <param name="httpClient"></param>
    /// <param name="log">Logger.</param>
    public MilvusMemoryStore(
        string host,
        int port,
        int vectorSize,
        string userName = "root",
        string password = "milvus",
        MilvusIndexType milvusIndexType = MilvusIndexType.AUTOINDEX,
        HttpClient? httpClient = null,
        ILogger? log = null)
    {
        this._vectorSize = vectorSize;
        this._milvusIndexType = milvusIndexType;
        this._log = log ?? NullLogger<MilvusMemoryStore>.Instance;
        this._milvusClient = new MilvusRestClient(host, port, userName, password, log: log, httpClient: httpClient);
    }

    ///<inheritdoc/>
    public async Task CreateCollectionAsync(
        string collectionName,
        CancellationToken cancellationToken = default)
    {
        if (!await this._milvusClient.HasCollectionAsync(
            collectionName,
            cancellationToken: cancellationToken).ConfigureAwait(false))
        {
            //Create collection
            await this._milvusClient.CreateCollectionAsync(collectionName,
                new FieldType[] {
                    FieldType.CreateVarchar(IdFieldName,maxLength: 100,isPrimaryKey: true),
                    FieldType.CreateFloatVector(EmbeddingFieldName,this._vectorSize),
                    FieldType.CreateVarchar(MetadataFieldName, maxLength: 1000) },
                cancellationToken: cancellationToken
                ).ConfigureAwait(false);

            //Create Index
            await this._milvusClient.CreateIndexAsync(
                collectionName,
                EmbeddingFieldName,
                Constants.DEFAULT_INDEX_NAME,
                this._milvusIndexType,
                MilvusMetricType.IP,
                null,
                cancellationToken: cancellationToken).ConfigureAwait(false);

            //Load Collection
            await this._milvusClient.LoadCollectionAsync(collectionName, cancellationToken: cancellationToken).ConfigureAwait(false);
        }
    }

    ///<inheritdoc/>
    public async Task DeleteCollectionAsync(
        string collectionName,
        CancellationToken cancellationToken = default)
    {
        await this._milvusClient.DropCollectionAsync(collectionName, cancellationToken).ConfigureAwait(false);
    }

    ///<inheritdoc/>
    public async Task<bool> DoesCollectionExistAsync(
        string collectionName,
        CancellationToken cancellationToken = default)
    {
        return await this._milvusClient.HasCollectionAsync(collectionName, cancellationToken: cancellationToken).ConfigureAwait(false);
    }

    ///<inheritdoc/>
    public async Task<MemoryRecord?> GetAsync(
        string collectionName,
        string key,
        bool withEmbedding,
        CancellationToken cancellationToken)
    {
        string expr = $"{IdFieldName} in [\"{key}\"]";
        MilvusQueryResult result = await this._milvusClient.QueryAsync(collectionName,
            expr,
            new[] { MetadataFieldName, EmbeddingFieldName },
            cancellationToken: cancellationToken).ConfigureAwait(false);

        if (result.FieldsData?.Any() != true || result.FieldsData.First().RowCount == 0)
        {
            return null;
        }

        var metadataField = result.FieldsData.First(p => p.FieldName == MetadataFieldName) as Field<string>;
        var embeddingField = result.FieldsData.First(p => p.FieldName == EmbeddingFieldName) as FloatVectorField;

        if (metadataField == null || embeddingField == null)
        {
            return null;
        }

        return MemoryRecord.FromJsonMetadata(
            metadataField.Data[0],
            new Embedding<float>(embeddingField.Data[0]));
    }

    public async IAsyncEnumerable<MemoryRecord> GetBatchAsync(
        string collectionName,
        IEnumerable<string> keys,
        bool withEmbeddings = false,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        List<string> keyList = keys.ToList();
        StringBuilder keyGroup = GetKeyGroup(keyList);
        string expr = $"{IdFieldName} in [{keyGroup}]";

        MilvusQueryResult result = await this._milvusClient.QueryAsync(collectionName,
            expr,
            new[] { MetadataFieldName, EmbeddingFieldName },
            cancellationToken: cancellationToken).ConfigureAwait(false);

        if (result.FieldsData?.Any() != true)
        {
            yield break;
        }

        var metadataField = result.FieldsData.First(p => p.FieldName == MetadataFieldName) as Field<string>;
        var embeddingField = result.FieldsData.First(p => p.FieldName == EmbeddingFieldName) as FloatVectorField;

        if (metadataField == null || embeddingField == null)
        {
            yield break;
        }

        for (int i = 0; i < metadataField.RowCount; i++)
        {
            yield return MemoryRecord.FromJsonMetadata(
                metadataField.Data[i],
                new Embedding<float>(embeddingField.Data[i]));
        }
    }

    ///<inheritdoc/>
    public async IAsyncEnumerable<string> GetCollectionsAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        IList<MilvusCollection> result = await this._milvusClient.ShowCollectionsAsync(cancellationToken: cancellationToken).ConfigureAwait(false);
        foreach (var collection in result)
        {
            yield return collection.CollectionName;
        }
    }

    ///<inheritdoc/>
    public async Task<(MemoryRecord, double)?> GetNearestMatchAsync(
        string collectionName,
        Embedding<float> embedding,
        double minRelevanceScore = 0,
        bool withEmbedding = false,
        CancellationToken cancellationToken = default)
    {
        //Milvus does not support vector field in out fields
        MilvusSearchResult searchResult = await this._milvusClient.SearchAsync(
            MilvusSearchParameters.Create(collectionName, EmbeddingFieldName, new[] { MetadataFieldName })
            .WithConsistencyLevel(MilvusConsistencyLevel.Strong)
            .WithTopK(topK: 1)
            .WithVectors(new[] { embedding.Vector.ToList() })
            .WithMetricType(MilvusMetricType.IP)
            .WithParameter("nprobe", "10"),
            cancellationToken: cancellationToken).ConfigureAwait(false);

        if (searchResult.Results.FieldsData.Any() != true || searchResult.Results.FieldsData.First().RowCount == 0)
        {
            return null;
        }

        double score = searchResult.Results.Scores[0];
        if (score < minRelevanceScore)
        {
            return null;
        }

        var metadataField = searchResult.Results.FieldsData[0] as Field<string>;
        if (metadataField == null)
        {
            return null;
        }

        if (withEmbedding)
        {
            var metadata = JsonSerializer.Deserialize<MemoryRecordMetadata>(metadataField.Data[0]);
            if (metadata == null)
            {
                return null;
            }

            MemoryRecord? nearestEmbedding = (await ((IMemoryStore)this).GetAsync(collectionName, metadata.Id, withEmbedding, cancellationToken: cancellationToken).ConfigureAwait(false));
            return nearestEmbedding == null ? null : (nearestEmbedding, score);
        }

        return (MemoryRecord.FromJsonMetadata(
            metadataField.Data[0],
            null), score);
    }

    ///<inheritdoc/>
    public async IAsyncEnumerable<(MemoryRecord, double)> GetNearestMatchesAsync(
        string collectionName,
        Embedding<float> embedding,
        int limit,
        double minRelevanceScore = 0,
        bool withEmbeddings = false,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        //Milvus does not support vector field in out fields
        MilvusSearchResult searchResult = await this._milvusClient.SearchAsync(
            MilvusSearchParameters.Create(collectionName, EmbeddingFieldName, new[] { MetadataFieldName })
            .WithConsistencyLevel(MilvusConsistencyLevel.Strong)
            .WithTopK(topK: limit)
            .WithVectors(new[] { embedding.Vector.ToList() })
            .WithMetricType(MilvusMetricType.IP)
            .WithParameter("nprobe", "10"),
            cancellationToken: cancellationToken)
            .ConfigureAwait(false);

        if (searchResult.Results.FieldsData.Any() != true || searchResult.Results.FieldsData.First().RowCount == 0)
        {
            yield break;
        }

        var metadataField = searchResult.Results.FieldsData[0] as Field<string>;
        if (metadataField == null)
        {
            yield break;
        }

        for (int i = 0; i < metadataField.RowCount; i++)
        {
            double score = searchResult.Results.Scores[i];
            if (score < minRelevanceScore)
            {
                continue;
            }

            if (withEmbeddings)
            {
                var metadata = JsonSerializer.Deserialize<MemoryRecordMetadata>(metadataField.Data[0]);
                if (metadata == null)
                {
                    continue;
                }

                MemoryRecord? nearestEmbedding = (await ((IMemoryStore)this).GetAsync(collectionName, metadata.Id, withEmbeddings, cancellationToken: cancellationToken).ConfigureAwait(false));
                if (nearestEmbedding != null)
                {
                    yield return (nearestEmbedding, score);
                }
            }
            else
            {
                yield return (MemoryRecord.FromJsonMetadata(
                    metadataField.Data[i],
                    null), score);
            }
        }
    }

    ///<inheritdoc/>
    public async Task RemoveAsync(
        string collectionName,
        string key,
        CancellationToken cancellationToken = default)
    {
        await this._milvusClient.DeleteAsync(
            collectionName,
            $"{IdFieldName} in [\"{key}\"]",
            cancellationToken: cancellationToken)
            .ConfigureAwait(false);
    }

    ///<inheritdoc/>
    public async Task RemoveBatchAsync(
        string collectionName,
        IEnumerable<string> keys,
        CancellationToken cancellationToken = default)
    {
        StringBuilder stringBuilder = GetKeyGroup(keys);

        await this._milvusClient.DeleteAsync(
            collectionName,
            $"{EmbeddingFieldName} in [{stringBuilder}]",
            cancellationToken: cancellationToken)
            .ConfigureAwait(false);
    }

    ///<inheritdoc/>
    public async Task<string> UpsertAsync(
        string collectionName,
        MemoryRecord record,
        CancellationToken cancellationToken = default)
    {
        //TODO: Check if record's id exists when autoId == false

        MilvusMutationResult insertResult = await this._milvusClient.InsertAsync(
            collectionName,
            new Field[] {
                this.ToIdField(record),
                this.ToFloatField(record),
                this.ToMetadataField(record)},
            cancellationToken: cancellationToken)
            .ConfigureAwait(false);

        return insertResult.Ids.IdField.StrId.Data[0];
    }

    ///<inheritdoc/>
    public async IAsyncEnumerable<string> UpsertBatchAsync(
        string collectionName,
        IEnumerable<MemoryRecord> records,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (records?.Any() != true)
        {
            yield break;
        }

        MilvusMutationResult insertResult = await this._milvusClient.InsertAsync(
            collectionName,
            new Field[] {
                this.ToIdField(records),
                this.ToFloatField(records),
                this.ToMetadataField(records)},
            cancellationToken: cancellationToken)
            .ConfigureAwait(false);

        foreach (var id in insertResult.Ids.IdField.StrId.Data)
        {
            yield return id;
        }
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!this._disposedValue)
        {
            if (disposing)
            {
                this._milvusClient?.Dispose();
            }
            this._disposedValue = true;
        }
    }

    public void Dispose()
    {
        this.Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    #region Private ===============================================================================
    private static StringBuilder GetKeyGroup(IEnumerable<string> keys)
    {
        StringBuilder stringBuilder = new();
        foreach (var key in keys)
        {
            if (stringBuilder.Length > 0)
            {
                stringBuilder.Append(", ");
            }
            stringBuilder.Append($"\"{key}\"");
        }

        return stringBuilder;
    }

    private Field<string> ToIdField(MemoryRecord record)
    {
        return Field.Create(IdFieldName, new[] { record.Metadata.Id });
    }

    private Field<string> ToIdField(IEnumerable<MemoryRecord> records)
    {
        return Field.Create(IdFieldName, records.Select(m => m.Metadata.Id).ToList());
    }

    private FloatVectorField ToFloatField(MemoryRecord record)
    {
        return Field.CreateFloatVector(EmbeddingFieldName, new List<List<float>> { record.Embedding.Vector.ToList() });
    }

    private FloatVectorField ToFloatField(IEnumerable<MemoryRecord> records)
    {
        return Field.CreateFloatVector(EmbeddingFieldName, records.Select(m => m.Embedding.Vector.ToList()).ToList());
    }

    private Field<string> ToMetadataField(MemoryRecord record)
    {
        return Field.CreateVarChar(MetadataFieldName, new[] { record.GetSerializedMetadata() });
    }

    private Field<string> ToMetadataField(IEnumerable<MemoryRecord> record)
    {
        return Field.CreateVarChar(MetadataFieldName, record.Select(m => m.GetSerializedMetadata()).ToList());
    }
    #endregion
}
