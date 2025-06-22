using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using System.IO;
using System.Linq;
using System.Dynamic;
using System.Reflection;
using Tensorflow; // Use actual TensorFlow.NET library
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using Accord.MachineLearning;
using Accord.Math.Distances;
using System.Text;
// Explicitly alias System.Numerics.Vector3 to avoid ambiguity if Accord.Math.Vector3 is also used or could be introduced.
// However, since Accord.Math.Vector3 is not directly used in a conflicting way in the snippets provided,
// we will use full qualification for System.Numerics.Vector3 where ambiguity arises.
// For clarity, if Accord.Math.Vector3 were used, aliasing would be:
// using AccordVector3 = Accord.Math.Vector3;
// using SNVector3 = System.Numerics.Vector3;
// For now, we will use System.Numerics.Vector3 directly or fully qualified.
using System.Numerics; // This will be System.Numerics.Vector3
using Accord.IO;
using AutoGen;
using AutoGen.Core;
using Accord.Math;
// Graph from Tensorflow will be Tensorflow.Graph
// Graph from AutoGen will be AutoGen.Core.Graph

namespace Agentic_Mod
{
    #region Data Structure Representations

    // This region defines the structures representing various data records used within the system.
    // These structures act as schemas for the simulated data storage and in-memory data handling,
    // serving as primary data dependencies for the processing logic.

    /// <summary>
    /// Represents the primary record storing outcomes of the ML initiation process.
    /// This is the central data entity being generated and updated throughout the workflow.
    /// </summary>
    public class CoreMlOutcomeRecord
    {
        // Properties define the fields of the core outcome record.
        // These fields hold identifiers, timestamps, categorical information,
        // serialized model data, and resulting vectors.
        // Operational Process Dependency: Created/retrieved by SequentialInitialProcessingUnitC (Factory One).
        // Updated by SequentialInitialProcessingUnitC (Factory One) with simulated model data.
        // Updated by SequentialFinalProcessingUnitD (Factory Four) with final vector results.
        // Referenced by ParallelProcessingUnitA (Factory Two) and ParallelProcessingUnitB (Factory Three).
        // Retrieved as the final output by the orchestrating console method.

        public int RecordIdentifier { get; set; } // Unique identifier for the record
        public int? AssociatedCustomerIdentifier { get; set; } // Link to a customer entity
        public DateTime? OutcomeGenerationTimestamp { get; set; } // Timestamp of when the outcome was generated/updated
        public int? CategoricalClassificationIdentifier { get; set; } // Identifier for a categorical classification
        public string? CategoricalClassificationDescription { get; set; } // Descriptive name for the classification
        public byte[]? SerializedSimulatedModelData { get; set; } // Binary data representing the trained simulated model (weights)
        public byte[]? AncillaryBinaryDataPayload { get; set; } // Additional binary data payload (bias)
        public string? DerivedProductFeatureVector { get; set; } // String representation of the resulting product vector
        public string? DerivedServiceBenefitVector { get; set; } // String representation of the resulting service vector
    }

    /// <summary>
    /// Represents contextual information about an associated customer.
    /// A supplementary record linked to the customer identifier.
    /// </summary>
    public class AssociatedCustomerContext
    {
        // Properties define the fields for customer context.
        // Operational Process Dependency: Created/retrieved by SequentialInitialProcessingUnitC (Factory One).
        // Referenced during the workflow for context.
        public int ContextIdentifier { get; set; }
        public string? CustomerPrimaryGivenName { get; set; }
        public string? CustomerFamilyName { get; set; }
        public string? CustomerContactPhoneNumber
        {
            get; set;
        }
        public string? CustomerStreetAddress { get; set; }
        public int? CustomerLinkIdentifier { get; set; } // Foreign key linking to a customer concept
        public string? AffiliatedCompanyName { get; set; }
    }

    /// <summary>
    /// Represents a specific work order associated with a customer.
    /// Another supplementary record linked to the customer and potentially an order identifier.
    /// </summary>
    public class OperationalWorkOrderRecord
    {
        // Properties define the fields for a work order record.
        // Operational Process Dependency: Created/retrieved by SequentialInitialProcessingUnitC (Factory One).
        // Referenced during the workflow for context.
        public int OrderRecordIdentifier { get; set; }
        public int? CustomerLinkIdentifier { get; set; }
        public int? SpecificOrderIdentifier { get; set; }
    }

    /// <summary>
    /// Represents an event or operation related to the ML initiation.
    /// Records actions or metadata during the process.
    /// </summary>
    public class MlInitialOperationEvent
    {
        // Properties define the fields for an operational event record.
        // Operational Process Dependency: Created/retrieved by SequentialInitialProcessingUnitC (Factory One).
        // Referenced during the workflow.
        public int EventIdentifier { get; set; }
        public int? CustomerLinkIdentifier { get; set; }
        public int? RelatedOrderIdentifier { get; set; }
        public int? InternalOperationIdentifier { get; set; }
        public byte[]? EventPayloadData { get; set; } // Binary data specific to the event
    }

    /// <summary>
    /// Represents data resulting from Quality Assurance (QA) checks on the ML outcome.
    /// Stores validation results related to the process.
    /// </summary>
    public class MlOutcomeValidationRecord
    {
        // Properties define the fields for a validation record.
        // Operational Process Dependency: Created/retrieved by SequentialInitialProcessingUnitC (Factory One).
        // Referenced during the workflow.
        public int ValidationRecordIdentifier { get; set; }
        public int? CustomerLinkIdentifier { get; set; }
        public int? RelatedOrderIdentifier { get; set; }
        public byte[]? ValidationResultData { get; set; } // Binary data containing QA results
    }

    /// <summary>
    /// Represents data collected or generated during the initial operational stage.
    /// Contains details about related products, services, and vectors for this stage.
    /// </summary>
    public class InitialOperationalStageData
    {
        // Properties define the fields for initial stage data.
        // Operational Process Dependency: Created/retrieved by SequentialInitialProcessingUnitC (Factory One).
        // May be used as input or context for ML steps.
        public int StageIdentifier { get; set; }
        public int? CustomerLinkIdentifier { get; set; }
        public int? RelatedOrderIdentifier { get; set; }
        public int? InternalOperationIdentifier { get; set; }
        public int? ProcessOperationalIdentifier { get; set; }
        public int? CustomerServiceOperationIdentifier { get; set; }
        public int? SalesProcessIdentifier { get; set; }
        public int? LinkedSubServiceA { get; set; } // Links to simulated service data
        public int? LinkedSubServiceB { get; set; } // Links to simulated service data
        public int? LinkedSubServiceC { get; set; } // Links to simulated service data
        public int? LinkedSubProductA { get; set; } // Links to simulated product data
        public int? LinkedSubProductB { get; set; } // Links to simulated product data
        public int? LinkedSubProductC { get; set; } // Links to simulated product data
        public string? StageSpecificData { get; set; } // String data for this stage
        public string? StageProductVectorSnapshot { get; set; } // Product vector state at this stage
        public string? StageServiceVectorSnapshot { get; set; } // Service vector state at this stage
    }

    #endregion

    #region Persistent And Transient Storage

    // This region defines static classes simulating data persistence and runtime memory management.
    // These components provide the data storage dependencies for the processing factories.

    /// <summary>
    /// Static class simulating persistent data storage (like a database) for testing.
    /// Holds various lists of data records.
    /// </summary>
    public static class InMemoryTestDataSet
    {
        // Holds static lists simulating database tables.
        // Operational Process Dependency: SequentialInitialProcessingUnitC reads from and writes to these lists.
        // InitiateMlOutcomeGeneration reads the final result from SimulatedCoreOutcomes.
        // GetOutcomeRecordByIdentifier and GetAllOutcomeRecords endpoints read from SimulatedCoreOutcomes.
        // The controller constructor populates RuntimeProcessingContext with sample product/service data from here.

        /// <summary>
        /// Simulated static list representing the collection of CoreMlOutcomeRecord entities.
        /// </summary>
        public static readonly List<CoreMlOutcomeRecord> SimulatedCoreOutcomes = new List<CoreMlOutcomeRecord>
        {
            // Initial dummy data representing existing records.
            // Operational Process Dependency: The source for retrieving existing records by SequentialInitialProcessingUnitC.
            // The destination for saving new or updated records by SequentialInitialProcessingUnitC and SequentialFinalProcessingUnitD.
            new CoreMlOutcomeRecord
            {
                RecordIdentifier = 1,
                AssociatedCustomerIdentifier = 123,
                OutcomeGenerationTimestamp = DateTime.UtcNow.AddDays(-10),
                CategoricalClassificationIdentifier = 5,
                CategoricalClassificationDescription = "Initial Historical Category",
                SerializedSimulatedModelData = new byte[] { 1, 2, 3, 4 }, // Placeholder binary data
                AncillaryBinaryDataPayload = new byte[] { 5, 6, 7, 8 }, // Placeholder binary data
                DerivedProductFeatureVector = "P_Vect_Init: X=1.0, Y=2.0, Z=3.0", // Sample vector data
                DerivedServiceBenefitVector = "S_Vect_Init: X=4.0, Y=5.0, Z=6.0" // Sample vector data
            },
            new CoreMlOutcomeRecord
            {
                RecordIdentifier = 2,
                AssociatedCustomerIdentifier = 456,
                OutcomeGenerationTimestamp = DateTime.UtcNow.AddDays(-1),
                CategoricalClassificationIdentifier = 10,
                CategoricalClassificationDescription = "Second Recent Category",
                SerializedSimulatedModelData = new byte[] { 9, 10, 11, 12 }, // Placeholder binary data
                AncillaryBinaryDataPayload = new byte[] { 13, 14, 15, 16 }, // Placeholder binary data
                DerivedProductFeatureVector = "P_Vect_Init: X=10.0, Y=11.0, Z=12.0", // Sample vector data
                DerivedServiceBenefitVector = "S_Vect_Init: X=13.0, Y=14.0, Z=15.0" // Sample vector data
            }
        };

        /// <summary>
        /// Simulated static list representing customer context records.
        /// </summary>
        public static readonly List<AssociatedCustomerContext> SimulatedCustomerContexts = new List<AssociatedCustomerContext>
        {
             // Initial dummy data.
            // Operational Process Dependency: SequentialInitialProcessingUnitC checks/adds records here, linked to the customer.
            new AssociatedCustomerContext
            {
                ContextIdentifier = 1,
                CustomerPrimaryGivenName = "John",
                CustomerFamilyName = "Doe",
                CustomerContactPhoneNumber = "555-1234-Sim",
                CustomerStreetAddress = "123 Main St Sim",
                CustomerLinkIdentifier = 123,
                AffiliatedCompanyName = "Acme Inc. Sim"
            },
            new AssociatedCustomerContext
            {
                ContextIdentifier = 2,
                CustomerPrimaryGivenName = "Jane",
                CustomerFamilyName = "Smith",
                CustomerContactPhoneNumber = "555-5678-Sim",
                CustomerStreetAddress = "456 Elm St Sim",
                CustomerLinkIdentifier = 456,
                AffiliatedCompanyName = "XYZ Corp Sim"
            }
        };

        /// <summary>
        /// Simulated static list representing operational work order records.
        /// </summary>
        public static readonly List<OperationalWorkOrderRecord> SimulatedWorkOrders = new List<OperationalWorkOrderRecord>
        {
            // Initial dummy data.
            // Operational Process Dependency: SequentialInitialProcessingUnitC checks/adds records here, linked to the customer/order.
            new OperationalWorkOrderRecord
            {
                OrderRecordIdentifier = 1,
                CustomerLinkIdentifier = 123,
                SpecificOrderIdentifier = 1001
            },
            new OperationalWorkOrderRecord
            {
                OrderRecordIdentifier = 2,
                CustomerLinkIdentifier = 456,
                SpecificOrderIdentifier = 1002
            }
        };

        /// <summary>
        /// Simulated static list representing ML initial operation events.
        /// </summary>
        public static readonly List<MlInitialOperationEvent> SimulatedOperationalEvents = new List<MlInitialOperationEvent>
        {
            // Initial dummy data.
            // Operational Process Dependency: SequentialInitialProcessingUnitC checks/adds records here, linked to the model initialization.
            new MlInitialOperationEvent
            {
                EventIdentifier = 1,
                CustomerLinkIdentifier = 123,
                RelatedOrderIdentifier = 1001,
                InternalOperationIdentifier = 5001,
                EventPayloadData = new byte[] { 21, 22, 23, 24 } // Placeholder binary data
            },
            new MlInitialOperationEvent
            {
                EventIdentifier = 2,
                CustomerLinkIdentifier = 456,
                RelatedOrderIdentifier = 1002,
                InternalOperationIdentifier = 5002,
                EventPayloadData = new byte[] { 25, 26, 27, 28 } // Placeholder binary data
            }
        };

        /// <summary>
        /// Simulated static list representing ML outcome validation records.
        /// </summary>
        public static readonly List<MlOutcomeValidationRecord> SimulatedOutcomeValidations = new List<MlOutcomeValidationRecord>
        {
            // Initial dummy data.
            // Operational Process Dependency: SequentialInitialProcessingUnitC checks/adds records here, linked to the model initialization.
            new MlOutcomeValidationRecord
            {
                ValidationRecordIdentifier = 1,
                CustomerLinkIdentifier = 123,
                RelatedOrderIdentifier = 1001,
                ValidationResultData = new byte[] { 31, 32, 33, 34 } // Placeholder binary data
            },
            new MlOutcomeValidationRecord
            {
                ValidationRecordIdentifier = 2,
                CustomerLinkIdentifier = 456,
                RelatedOrderIdentifier = 1002,
                ValidationResultData = new byte[] { 35, 36, 37, 38 } // Placeholder binary data
            }
        };

        /// <summary>
        /// Simulated static list representing initial operational stage data records.
        /// </summary>
        public static readonly List<InitialOperationalStageData> SimulatedInitialOperationalStages = new List<InitialOperationalStageData>
        {
            // Initial dummy data.
            // Operational Process Dependency: SequentialInitialProcessingUnitC checks/adds records here, linked to the operation stage.
            new InitialOperationalStageData
            {
                StageIdentifier = 1,
                CustomerLinkIdentifier = 123,
                RelatedOrderIdentifier = 1001,
                InternalOperationIdentifier = 5001,
                ProcessOperationalIdentifier = 6001,
                CustomerServiceOperationIdentifier = 7001,
                SalesProcessIdentifier = 8001,
                LinkedSubServiceA = 1, // Links to SampleServiceOfferings
                LinkedSubServiceB = 2, // Links to SampleServiceOfferings
                LinkedSubServiceC = 3, // Links to SampleServiceOfferings
                LinkedSubProductA = 1, // Links to SampleProductInventory
                LinkedSubProductB = 2, // Links to SampleProductInventory
                LinkedSubProductC = 3, // Links to SampleProductInventory
                StageSpecificData = "Sample data for initial operational stage record 1",
                StageProductVectorSnapshot = "Stage1_P_Vect: X=1.0, Y=2.0, Z=3.0",
                StageServiceVectorSnapshot = "Stage1_S_Vect: X=4.0, Y=5.0, Z=6.0"
            },
            new InitialOperationalStageData
            {
                StageIdentifier = 2,
                CustomerLinkIdentifier = 456,
                RelatedOrderIdentifier = 1002,
                InternalOperationIdentifier = 5002,
                ProcessOperationalIdentifier = 6002,
                CustomerServiceOperationIdentifier = 7002,
                SalesProcessIdentifier = 8002,
                LinkedSubServiceA = 7, // Links to SampleServiceOfferings
                LinkedSubServiceB = 8, // Links to SampleServiceOfferings
                LinkedSubServiceC = 9, // Links to SampleServiceOfferings
                LinkedSubProductA = 10, // Links to SampleProductInventory
                LinkedSubProductB = 11, // Links to SampleProductInventory
                LinkedSubProductC = 12, // Links to SampleProductInventory
                StageSpecificData = "Sample data for initial operational stage record 2",
                StageProductVectorSnapshot = "Stage1_P_Vect: X=10.0, Y=11.0, Z=12.0",
                StageServiceVectorSnapshot = "Stage1_S_Vect: X=13.0, Y=14.0, Z=15.0"
            }
        };

        /// <summary>
        /// Simulated product data entries for testing.
        /// Represents a separate dataset linked by ID.
        /// </summary>
        public static readonly List<dynamic> SampleProductInventory = new List<dynamic>
        {
            // Dummy product data.
            // Operational Process Dependency: Loaded into RuntimeProcessingContext by the controller constructor.
            // Could potentially be used by processing units to retrieve details based on LinkedSubProduct IDs in InitialOperationalStageData.
            new { Identifier = 1, ItemDesignation = "Product A Alpha", Categorization = "Type 1 Assembly", QuantityAvailable = 10, MonetaryValue = 99.99, CostContributionValue = 0.15 },
            new { Identifier = 2, ItemDesignation = "Product B Beta", Categorization = "Type 2 Component", QuantityAvailable = 20, MonetaryValue = 149.99, CostContributionValue = 0.25 },
            new { Identifier = 3, ItemDesignation = "Product C Gamma", Categorization = "Type 3 Module", QuantityAvailable = 15, MonetaryValue = 199.99, CostContributionValue = 0.20 }
        };

        /// <summary>
        /// Simulated service data entries for testing.
        /// Represents a separate dataset linked by ID.
        /// </summary>
        public static readonly List<dynamic> SampleServiceOfferings = new List<dynamic>
        {
            // Dummy service data.
            // Operational Process Dependency: Loaded into RuntimeProcessingContext by the controller constructor.
            // Could potentially be used by processing units to retrieve details based on LinkedSubService IDs in InitialOperationalStageData.
            new { Identifier = 1, ServiceNameDescriptor = "Service A Alpha", Categorization = "Tier 1 Support", FulfillmentQuantity = 5, MonetaryValue = 299.99, CostContributionValue = 0.30 },
            new { Identifier = 2, ServiceNameDescriptor = "Service B Beta", Categorization = "Tier 2 Consulting", FulfillmentQuantity = 10, MonetaryValue = 399.99, CostContributionValue = 0.35 },
            new { Identifier = 3, ServiceNameDescriptor = "Service C Gamma", Categorization = "Tier 3 Managed", FulfillmentQuantity = 8, MonetaryValue = 599.99, CostContributionValue = 0.40 }
        };
    }

    /// <summary>
    /// Static class simulating transient, runtime memory storage accessible within a request context.
    /// Uses a dynamic object to hold properties added during the request lifecycle.
    /// </summary>
    public static class RuntimeProcessingContext
    {
        // Provides static, request-scoped (conceptually, though globally static here) storage using ExpandoObject.
        // Operational Process Dependency: SequentialInitialProcessingUnitC stores references to created/found records and intermediate data here.
        // The controller constructor stores initial product/service data here.
        // This allows different processing units (factories) to share state and data during the execution of a single API call.
        private static readonly ExpandoObject _volatileKeyValueStore = new ExpandoObject();
        private static readonly dynamic _dynamicAccessView = _volatileKeyValueStore;
        private static RuntimeMethodHandle _cachedMethodHandle; // Not directly used in the core ML flow demonstrated

        /// <summary>
        /// Adds or updates a value associated with a string key in the runtime storage.
        /// </summary>
        /// <param name="keyDescriptor">The string descriptor for the value</param>
        /// <param name="contextValue">The value to store</param>
        public static void StoreContextValue(string keyDescriptor, object contextValue)
        {
            // Stores data for later retrieval in the same request.
            // Operational Process Dependency: Called by SequentialInitialProcessingUnitC to store references to records and simulated model data.
            // Called by the controller constructor to store initial data from InMemoryTestDataSet.
            var dictionary = (IDictionary<string, object>)_volatileKeyValueStore;
            dictionary[keyDescriptor] = contextValue;
        }

        /// <summary>
        /// Retrieves a value from the runtime storage using its string key descriptor.
        /// Returns null if the key descriptor does not exist.
        /// </summary>
        /// <param name="keyDescriptor">The string descriptor for the value</param>
        /// <returns>The stored value, or null if the key was not found</returns>
        public static object? RetrieveContextValue(string keyDescriptor)
        {
            // Retrieves data stored earlier in the same request.
            // Operational Process Dependency: Called by SequentialInitialProcessingUnitC for verification and potentially by other units
            // if they needed data stored by SequentialInitialProcessingUnitC (e.g., retrieving the CoreMlOutcomeRecord).
            var dictionary = (IDictionary<string, object>)_volatileKeyValueStore;
            return dictionary.TryGetValue(keyDescriptor, out var value) ? value : null;
        }

        /// <summary>
        /// Gets a dynamic view of the runtime storage for flexible access.
        /// </summary>
        public static dynamic DynamicContextView => _dynamicAccessView;
        // Provides direct dynamic access to the underlying ExpandoObject.


        /// <summary>
        /// Gets all keys currently stored in the runtime context.
        /// </summary>
        /// <returns>Collection of all context keys</returns>
        public static IEnumerable<string> GetAllRuntimeContextKeys()
        {
            var dictionary = (IDictionary<string, object>)_volatileKeyValueStore;
            return dictionary.Keys;
        }
        /// <summary>
        /// Sets a cached method handle (not directly used in ML flow).
        /// </summary>
        /// <param name="handle">The RuntimeMethodHandle to cache</param>
        public static void CacheMethodHandle(RuntimeMethodHandle handle)
        {
            _cachedMethodHandle = handle;
        }
        // Not directly involved in the core ML process flow demonstrated.

        /// <summary>
        /// Gets the stored cached method handle (not directly used in ML flow).
        /// </summary>
        /// <returns>The RuntimeMethodHandle</returns>
        public static RuntimeMethodHandle RetrieveCachedMethodHandle()
        {
            return _cachedMethodHandle;
        }
        // Not directly involved in the core ML process flow demonstrated.
    }

    /// <summary>
    /// Component intended for orchestrating concurrent processing tasks.
    /// Manages shared resources for parallel operations (currently minimal).
    /// </summary>
    public class ConcurrentOperationManager
    {
        // Intended for managing shared state or coordination for parallel tasks.
        // Operational Process Dependency: An instance is held by the controller. Its ResetSharedDataStore method is available (though not explicitly called in this flow).
        // The ConcurrentDictionary within it *could* be used for thread-safe sharing, similar to the results dictionaries used in ParallelProcessingUnitA/B/SequentialFinalProcessingUnitD.
        private readonly ConcurrentDictionary<string, object> _parallelSharedDataStore = new ConcurrentDictionary<string, object>();

        /// <summary>
        /// Clears data within the shared data store.
        /// </summary>
        public void ResetSharedDataStore()
        {
            _parallelSharedDataStore.Clear();
        }
    }

    #endregion


    // Defines the class that orchestrates the entire ML outcome generation process.
    // Adapted from the ASP.NET Core Controller.
    public class MlProcessOrchestrator
    {
        private static int _requestSessionSequenceCounter = 0; // Counter for generating unique request session IDs
        private readonly ConcurrentDictionary<int, Tensorflow.Session> _activeMlSessions; // Tracks active actual ML sessions by unique ID
        private readonly ConcurrentOperationManager _operationConcurrencyManager; // Instance of the concurrency manager

        /// <summary>
        /// Constructor for the ML process orchestration class.
        /// Initializes session tracking and the concurrency manager.
        /// Populates the runtime processing context with initial simulated data.
        /// </summary>
        public MlProcessOrchestrator()
        {
            // Initializes the concurrent dictionary for tracking actual ML sessions and the concurrency manager.
            _activeMlSessions = new ConcurrentDictionary<int, Tensorflow.Session>();
            _operationConcurrencyManager = new ConcurrentOperationManager();

            // Initialize runtime memory with simulated data from the test dataset.
            // This makes simulated product/service data easily accessible to processing logic via RuntimeProcessingContext.
            // Operational Process Dependency: Data loaded here can be retrieved via RuntimeProcessingContext.RetrieveContextValue
            // by processing units that need lookups based on IDs from InitialOperationalStageData.
            RuntimeProcessingContext.StoreContextValue("All_Simulated_Service_Offerings", InMemoryTestDataSet.SampleServiceOfferings);
            RuntimeProcessingContext.StoreContextValue("All_Simulated_Product_Inventory", InMemoryTestDataSet.SampleProductInventory);

            // Configure TensorFlow.NET (equivalent of disabling eager execution in older TF versions)
            // tf.compat.v1.disable_eager_execution(); // Already non-eager by default for graph mode
        }

        internal static void DisposeGraphAndSession(ref Tensorflow.Graph graph, ref Tensorflow.Session session)
        {
            try
            {
                session?.Dispose();
                session = null;

                // Graph disposal - manual cleanup for TensorFlow.NET 0.150.0
                graph = null;
            }
            catch (Exception ex)
            {

            }
        }


        /// <summary>
        /// This method initiates the machine learning outcome generation process for a specific customer.
        /// It orchestrates a sequence involving initial setup, parallel processing, and final aggregation,
        /// using simulated services and actual TensorFlow.NET operations for ML and data persistence simulation.
        /// </summary>
        /// <param name="customerIdentifier">The unique identifier of the customer for whom to perform the ML outcome generation.</param>
        /// <returns>The final CoreMlOutcomeRecord representing the results of the ML outcome generation.</returns>
        /// <exception cref="ArgumentNullException">Thrown if customerIdentifier is null.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the workflow fails at a critical step.</exception>
        /// <exception cref="Exception">Catches and re-throws unexpected exceptions during the workflow.</exception>
        public async Task<CoreMlOutcomeRecord> InitiateMlOutcomeGeneration(int customerIdentifier)
        {
            // This method orchestrates the entire multi-step workflow.

            /// <summary>
            /// Operational Step 1: Input Validation and Workflow Initialization
            /// </summary>
            // Validate the mandatory customerIdentifier parameter.
            // Operational Process Dependency: This is the initial input gate. Prevents any further processing if invalid.
            // If validation passes, proceeds to set up the request context and session tracking.
            // Since customerIdentifier is now a non-nullable int parameter, the check `!customerIdentifier.HasValue` is replaced
            // by handling potential issues related to the value's meaning (e.g., 0 or negative if not intended).
            // For this conversion, we'll assume any positive int is valid input. If 0 is passed, the downstream logic
            // might still function, but we can add a check if needed. Let's add a check for <= 0.
            if (customerIdentifier <= 0)
            {
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Error: Customer identifier must be a positive integer to initiate ML outcome generation.");
                throw new ArgumentOutOfRangeException(nameof(customerIdentifier), "Customer identifier must be a positive integer.");
            }


            // Generate a unique identifier for tracking this specific workflow execution request.
            // Operational Process Dependency: Used for logging and uniquely identifying actual ML sessions created for this request in the _activeMlSessions dictionary.
            var requestSequenceIdentifier = Interlocked.Increment(ref _requestSessionSequenceCounter);
            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Starting ML Outcome Generation Workflow Session {requestSequenceIdentifier} for customer {customerIdentifier}");

            // Create dedicated TensorFlow.NET sessions for the parallel processing units (Units A and B).
            // These are functional dependencies for ParallelProcessingUnitA and ParallelProcessingUnitB.
            // Note: Creating Session objects is resource-intensive. In a production scenario, consider pooling sessions or using a different TF.NET pattern.
            // These sessions are now vestigial if Units A and B manage their own graphs and sessions internally.
            // However, they are still created and disposed as per original structure.
            Tensorflow.Session? modelAProcessingSession_Orchestrator = null;
            Tensorflow.Session? modelBProcessingSession_Orchestrator = null;
            CoreMlOutcomeRecord? outcomeRecordAfterStepOne = null;
            CoreMlOutcomeRecord? finalOutcomeRecord = null;


            try
            {
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Preparing resources for parallel operations.");

                /// <summary>
                /// Operational Step 2: Prepare Resources for Processing Units
                /// </summary>
                // Initialize a container for the core outcome record and thread-safe storage for results from parallel units.
                // 'currentOutcomeRecord' will initially be an empty object reference; SequentialInitialProcessingUnitC will establish the actual record instance.
                // 'modelAConcurrentResults' and 'modelBConcurrentResults' are used by ParallelProcessingUnitA and ParallelProcessingUnitB to return results to SequentialFinalProcessingUnitD.
                // Operational Process Dependency: These objects are created here and passed as parameters to the subsequent processing unit methods to facilitate data flow and result aggregation.
                var currentOutcomeRecord = new CoreMlOutcomeRecord(); // Container for the primary record being processed
                var modelAConcurrentResults = new ConcurrentDictionary<string, object>(); // Thread-safe store for results from Unit A
                var modelBConcurrentResults = new ConcurrentDictionary<string, object>(); // Thread-safe store for results from Unit B


                // Create actual TensorFlow.NET sessions for the parallel processing units (Units A and B).
                // Using separate graphs/sessions for independent parallel tasks is common.
                // The Session itself *is* disposable and is handled in the finally block.
                // These sessions are passed to Units A and B, but if those units create their own
                // graphs and sessions internally per the fix, these specific session instances
                // might not be used for the core TF operations within those units.
                modelAProcessingSession_Orchestrator = tf.Session(tf.Graph()); // Pass graph to session constructor
                modelBProcessingSession_Orchestrator = tf.Session(tf.Graph()); // Pass graph to session constructor


                /// <summary>
                /// Operational Step 3: Register Actual ML Sessions for Management
                /// </summary>
                // Register the created actual ML sessions in the controller's session manager using unique IDs derived from the requestSequenceIdentifier.
                // Operational Process Dependency: This is necessary for proper resource disposal in the 'finally' block at the end of the workflow.
                _activeMlSessions.TryAdd(requestSequenceIdentifier * 2, modelAProcessingSession_Orchestrator);     // Even numbered ID for Unit A session
                _activeMlSessions.TryAdd(requestSequenceIdentifier * 2 + 1, modelBProcessingSession_Orchestrator); // Odd numbered ID for Unit B session

                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Executing Sequential Initial Processing Unit (C).");

                /// <summary>
                /// Operational Step 4 (Sequential): Execute Initial Processing Unit (C)
                /// </summary>
                // Execute SequentialInitialProcessingUnitC (actual "ProcessFactoryOne"). This step runs sequentially first.
                // This unit is responsible for creating or retrieving the core CoreMlOutcomeRecord for the customer and establishing associated dependency records in simulated persistence if necessary, or loading existing ones.
                // It performs actual ML training (Model C) using TensorFlow.NET and saves the resulting model data (weights/biases).
                // Operational Process Dependency: Requires the initial 'currentOutcomeRecord' container, the customerIdentifier, and the requestSequenceIdentifier for context and logging.
                // Internally depends on InMemoryTestDataSet and RuntimeProcessingContext.
                // Subsequent Usage: The successful completion of this unit and its saved model data are dependencies for the parallel processing units (A and B) as they may use this model for inference/further processing.
                // It populates InMemoryTestDataSet and RuntimeProcessingContext with the initial/updated CoreMlOutcomeRecord and related data.
                await SequentialInitialProcessingUnitC(currentOutcomeRecord, customerIdentifier, requestSequenceIdentifier);

                // Retrieve the CoreMlOutcomeRecord object from simulated persistence *after* SequentialInitialProcessingUnitC has potentially created or updated it.
                // This ensures the orchestrator method has the latest state of the record before passing it to parallel units.
                // Operational Process Dependency: Depends on SequentialInitialProcessingUnitC successfully creating/findling and adding/updating the record in InMemoryTestDataSet.
                // Subsequent Usage: The 'outcomeRecordAfterStepOne' reference is used to check if Step 4 was successful and is then assigned to 'currentOutcomeRecord' to be passed to parallel units.
                outcomeRecordAfterStepOne = InMemoryTestDataSet.SimulatedCoreOutcomes
                                        .FirstOrDefault(r => r.AssociatedCustomerIdentifier == customerIdentifier);

                // Check if SequentialInitialProcessingUnitC successfully established the core outcome record.
                // Operational Process Dependency: This check depends on the result of the retrieval attempt after Step 4.
                // Subsequent Usage: If the record is null, the workflow cannot continue, and an error is returned. Otherwise, the workflow proceeds to the parallel steps.
                if (outcomeRecordAfterStepOne != null)
                {
                    currentOutcomeRecord = outcomeRecordAfterStepOne; // Update the local reference to the retrieved record
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Core outcome record established successfully by Unit C (ID: {currentOutcomeRecord.RecordIdentifier}). Proceeding to parallel units.");
                }
                else
                {
                    // Error handling if Step 4 failed to establish the record.
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Sequential Initial Processing Unit C failed to create/find CoreMlOutcomeRecord for customer {customerIdentifier}. Workflow cannot proceed.");
                    throw new InvalidOperationException($"Failed to establish initial model data for customer {customerIdentifier}. Cannot start parallel processing units.");
                }

                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Starting Parallel Processing Units (A and B).");

                /// <summary>
                /// Operational Step 5 (Parallel): Execute Parallel Processing Units (A and B)
                /// </summary>
                // Execute ParallelProcessingUnitA (actual "ProcessFactoryTwo") and ParallelProcessingUnitB (actual "ProcessFactoryThree") concurrently using Task.WhenAll.
                // Operational Process Dependency: Both units depend on the core 'currentOutcomeRecord' object established by SequentialInitialProcessingUnitC (Step 4).
                // They also depend on their respective allocated actual ML sessions (modelAProcessingSession, modelBProcessingSession) and thread-safe result dictionaries (modelAConcurrentResults, modelBConcurrentResults).
                // Subsequent Usage: The main workflow waits here until both parallel tasks complete. Their outputs are stored in 'modelAConcurrentResults' and 'modelBConcurrentResults'.
                // Create dedicated task variables for clearer debugging and monitoring
                // Create dedicated task variables for clearer debugging and monitoring
                // Pass the orchestrator-created sessions; Units A and B will use internal sessions for TF ops if the fix is applied as interpreted.
                Task unitATask = ParallelProcessingUnitA(currentOutcomeRecord, customerIdentifier, requestSequenceIdentifier, modelAProcessingSession_Orchestrator, modelAConcurrentResults);
                Task unitBTask = ParallelProcessingUnitB(currentOutcomeRecord, customerIdentifier, requestSequenceIdentifier, modelBProcessingSession_Orchestrator, modelBConcurrentResults);

                try
                {
                    // Use Task.WhenAll to wait for both tasks to complete
                    await Task.WhenAll(unitATask, unitBTask);

                    // Add verification after tasks complete
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Parallel tasks completed. Verifying results dictionary contents:");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] - Model A has predictions: {modelAConcurrentResults.ContainsKey("ModelAPredictionsFlat")}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] - Model B has predictions: {modelBConcurrentResults.ContainsKey("ModelBPredictionsFlat")}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] - Model A outcome stored: {modelAConcurrentResults.ContainsKey("ModelAProcessingOutcome")}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] - Model B outcome stored: {modelBConcurrentResults.ContainsKey("ModelBProcessingOutcome")}");

                }
                catch (Exception ex)
                {
                    // Log detailed error information
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Error during parallel processing: {ex.Message}");

                    // Check individual task statuses
                    if (unitATask.IsFaulted)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Model A task faulted with error: {unitATask.Exception?.InnerException?.Message ?? "Unknown error"}");
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Model A stack trace: {unitATask.Exception?.InnerException?.StackTrace ?? "Not available"}");
                    }

                    if (unitBTask.IsFaulted)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Model B task faulted with error: {unitBTask.Exception?.InnerException?.Message ?? "Unknown error"}");
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Model B stack trace: {unitBTask.Exception?.InnerException?.StackTrace ?? "Not available"}");
                    }

                    // Continue with what data we have - unit D should handle partial results
                    if (!unitATask.IsFaulted || !unitBTask.IsFaulted)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Continuing with partial results from successful parallel tasks");
                    }
                    else
                    {
                        // Re-throw if both failed
                        throw;
                    }
                }


                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Parallel Processing Units A and B completed. Starting Sequential Final Processing Unit (D).");

                /// <summary>
                /// Operational Step 6 (Sequential): Execute Final Processing Unit (D)
                /// </summary>
                // Execute SequentialFinalProcessingUnitD (actual "ProcessFactoryFour"). This step runs sequentially after the parallel units have completed.
                // It is intended to combine or use the results from the parallel units to perform final calculations or updates to the core outcome record.
                // Operational Process Dependency: Depends on the 'currentOutcomeRecord' object (established/updated by Step 4).
                // Crucially depends on the results gathered by ParallelProcessingUnitA and ParallelProcessingUnitB (passed via modelAConcurrentResults and modelBConcurrentResults).
                // Depends on InMemoryTestDataSet for saving the final state of the outcome record.
                // Subsequent Usage: Finalizes the state of the CoreMlOutcomeRecord before it's retrieved and returned by the orchestrating method.
                await SequentialFinalProcessingUnitD(currentOutcomeRecord, customerIdentifier, requestSequenceIdentifier, modelAConcurrentResults, modelBConcurrentResults);

                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: ML Outcome Generation workflow completed successfully.");

                /// <summary>
                /// Operational Step 7 (Sequential): Retrieve Final Outcome and Return
                /// </summary>
                // Retrieve the final CoreMlOutcomeRecord from the simulated database after all processing steps have completed and been saved.
                // Operational Process Dependency: Depends on SequentialFinalProcessingUnitD having successfully updated the record in InMemoryTestDataSet.
                // Subsequent Usage: This is the final output returned by the API endpoint.
                finalOutcomeRecord = InMemoryTestDataSet.SimulatedCoreOutcomes
                                    .FirstOrDefault(r => r.AssociatedCustomerIdentifier == customerIdentifier);

                // Check if the final record was found (should be, if steps 4-6 succeeded and saved).
                // Operational Process Dependency: Depends on the result of the final retrieval attempt.
                // Subsequent Usage: Determines whether to return a success response with the data or an error if the final state isn't retrievable.
                if (finalOutcomeRecord != null)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Returning final CoreMlOutcomeRecord (ID: {finalOutcomeRecord.RecordIdentifier}) for customer {customerIdentifier}.");
                    return finalOutcomeRecord; // Return the final record on success
                }
                else
                {
                    // Fallback error if the final record isn't found in simulated storage, despite previous steps seemingly succeeding.
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Final CoreMlOutcomeRecord not found for customer {customerIdentifier} after implementation. Potential data saving issue.");
                    throw new InvalidOperationException("ML outcome generation completed, but the final outcome record could not be retrieved.");
                }

            }
            /// <summary>
            /// Operational Step 8 (Workflow Cleanup and Error Handling)
            /// </summary>
            // Catch any exceptions thrown during the orchestration or within the processing units.
            // Operational Process Dependency: Catches errors from validation, resource setup, or any of the called processing unit methods (which might re-throw).
            // Subsequent Usage: Logs the error and re-throws it to be caught by the calling code (e.g., Program.Main).
            catch (Exception ex)
            {
                // Log detailed error information.
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Unhandled Error during ML Outcome Generation Workflow: {ex.Message}");
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Stack Trace: {ex.StackTrace}");
                throw; // Re-throw the exception
            }
            finally
            {
                // Cleanup: Remove and dispose the actual ML sessions created by the orchestrator from the manager.
                // These are modelAProcessingSession_Orchestrator and modelBProcessingSession_Orchestrator.
                // Sessions created *inside* the processing units (e.g., SequentialInitialProcessingUnitC, 
                // ParallelProcessingUnitA, ParallelProcessingUnitB due to fixes) are disposed of in their respective finally blocks.
                if (_activeMlSessions.TryRemove(requestSequenceIdentifier * 2, out var sessionA))
                    sessionA?.Dispose();
                if (_activeMlSessions.TryRemove(requestSequenceIdentifier * 2 + 1, out var sessionB))
                    sessionB?.Dispose();

                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Associated actual ML session resources (orchestrator-level) cleaned up.");
            }
        }
































        /// <summary>
        /// Processes data for Model C (SequentialInitialProcessingUnitC).
        /// This is the *first sequential* processing step in the workflow.
        /// It is responsible for ensuring the core CoreMlOutcomeRecord exists for the customer,
        /// creating it and associated dependency records in simulated persistence if necessary, or loading existing ones.
        /// It performs actual machine learning training using TensorFlow.NET (Model C) and saves the resulting model data.
        /// </summary>
        /// <param name="outcomeRecord">A reference to the CoreMlOutcomeRecord container/instance to work with.</param>
        /// <param name="customerIdentifier">The customer identifier.</param>
        /// <param name="requestSequenceIdentifier">The request session identifier.</param>
        private async Task SequentialInitialProcessingUnitC(CoreMlOutcomeRecord outcomeRecord, int customerIdentifier, int requestSequenceIdentifier)
        {
            // Declare variables outside the try block to ensure they are accessible in catch/finally
            CoreMlOutcomeRecord? retrievedOrNewOutcomeRecord = null;
            bool isNewRecord = false; // Declared outside try block
            Tensorflow.Session? mlSession = null; // Local session for this sequential unit
            Tensorflow.Graph? graph = null; // Graph object

            // Added null forgiving operator (!) where needed based on flow analysis,
            // especially after retrievedOrNewOutcomeRecord is known to be non-null.
            RuntimeProcessingContext.StoreContextValue("SequentialProcessingUnitC_ActiveStatus", true);
            bool isActiveStart = (bool)RuntimeProcessingContext.RetrieveContextValue("SequentialProcessingUnitC_ActiveStatus")!; // Use null forgiving
            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: SequentialProcessingUnitC ActiveStatus property value: {isActiveStart}");
            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Starting Sequential Initial Processing Unit C (Actual Model C).");

            // Disable eager execution before defining any TensorFlow operations for graph mode
            // This needs to be done early in the application lifecycle, preferably once.
            // Doing it per method might have unintended consequences or be inefficient.
            // Assuming for the scope of this method modification that repeating it here is intended
            // per the original code structure, despite best practices.
            tf.compat.v1.disable_eager_execution();
            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Disabled eager execution for TensorFlow operations.");


            // Helper method definitions (included within this method as per constraint)
            #region Helper Methods (Required by this method)

            /// <summary>
            /// Transforms word-based samples into numerical embeddings using a simplified embedding technique.
            /// </summary>
            float[][] TransformWordsToEmbeddings(string[] wordSamples)
            {
                if (wordSamples == null) return new float[0][]; // Added null check
                int embeddingDimensions = 10; // Fixed embedding dimension
                float[][] embeddings = new float[wordSamples.Length][];

                for (int i = 0; i < wordSamples.Length; i++)
                {
                    embeddings[i] = new float[embeddingDimensions];
                    if (wordSamples[i] == null) continue; // Skip null samples

                    string[] words = wordSamples[i].Split(' ');

                    for (int j = 0; j < words.Length; j++)
                    {
                        string word = words[j];
                        if (string.IsNullOrEmpty(word)) continue; // Skip empty words
                        int hashBase = word.GetHashCode();
                        for (int k = 0; k < embeddingDimensions; k++)
                        {
                            int valueInt = Math.Abs(hashBase * (k + 1) * (j + 1) * 31);
                            float value = (valueInt % 1000) / 1000.0f;
                            embeddings[i][k] += value * (1.0f / (j + 1.0f));
                        }
                    }

                    float magnitudeSq = 0;
                    for (int k = 0; k < embeddingDimensions; k++) magnitudeSq += embeddings[i][k] * embeddings[i][k];
                    float magnitude = (float)Math.Sqrt(magnitudeSq);
                    if (magnitude > 1e-6f)
                    {
                        for (int k = 0; k < embeddingDimensions; k++) embeddings[i][k] /= magnitude;
                    }
                }
                return embeddings;
            }

            /// <summary>
            /// Converts a jagged array to a multidimensional array.
            /// </summary>
            float[,] ConvertJaggedToMultidimensional(float[][] jaggedArray)
            {
                if (jaggedArray == null || jaggedArray.Length == 0 || jaggedArray.Any(row => row == null))
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper: ConvertJaggedToMultidimensional received null, empty, or jagged array with null rows. Returning empty multidimensional array.");
                    return new float[0, 0];
                }
                if (jaggedArray[0].Length == 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper: ConvertJaggedToMultidimensional received jagged array with zero columns. Returning empty multidimensional array.");
                    return new float[jaggedArray.Length, 0];
                }

                int rows = jaggedArray.Length;
                int cols = jaggedArray[0].Length;
                float[,] result = new float[rows, cols];
                for (int i = 0; i < rows; i++)
                {
                    if (jaggedArray[i].Length != cols)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: Row {i} in jagged array has inconsistent length ({jaggedArray[i].Length} vs {cols}). Returning partial result for this row.");
                        int currentCols = jaggedArray[i].Length;
                        // Copy as many elements as possible up to the minimum of actual row length and expected column count
                        for (int j = 0; j < Math.Min(cols, currentCols); j++)
                        {
                            result[i, j] = jaggedArray[i][j];
                        }
                        // The rest of the row in 'result' will remain default(float), which is 0.0f
                    }
                    else
                    {
                        System.Buffer.BlockCopy(jaggedArray[i], 0, result, i * cols * sizeof(float), cols * sizeof(float));
                    }
                }
                return result;
            }

            /// <summary>
            /// Extracts a batch from a multidimensional array using indices.
            /// </summary>
            float[,] ExtractBatch(float[,] data, int[] batchIndices, int startIdx, int count)
            {
                if (data == null || batchIndices == null || data.GetLength(0) == 0 || batchIndices.Length == 0 || count <= 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper: ExtractBatch received invalid input data, indices, or count ({count}). Returning empty batch.");
                    return new float[0, data?.GetLength(1) ?? 0]; // Return empty batch with correct column count if data is not null
                }
                // Corrected bounds check logic to be less strict initially but check actual access
                if (startIdx < 0 || startIdx >= batchIndices.Length || startIdx + count > batchIndices.Length)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: Batch indices out of bounds for ExtractBatch (startIdx: {startIdx}, count: {count}, indices length: {batchIndices.Length}). Returning empty batch.");
                    return new float[0, data.GetLength(1)];
                }


                int cols = data.GetLength(1);
                float[,] batch = new float[count, cols];
                for (int i = 0; i < count; i++)
                {
                    int srcIdx = batchIndices[startIdx + i];
                    if (srcIdx < 0 || srcIdx >= data.GetLength(0))
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Error: Invalid index {srcIdx} accessed in source data for ExtractBatch at batch index {i}. Returning partial batch extracted so far.");
                        // Return partial batch extracted so far
                        var partialBatch = new float[i, cols];
                        System.Buffer.BlockCopy(batch, 0, partialBatch, 0, i * cols * sizeof(float));
                        return partialBatch;
                    }
                    System.Buffer.BlockCopy(data, srcIdx * cols * sizeof(float), batch, i * cols * sizeof(float), cols * sizeof(float));
                }
                return batch;
            }

            /// <summary>
            /// Shuffles an array randomly.
            /// </summary>
            void ShuffleArray(int[] shuffleIndices)
            {
                if (shuffleIndices == null) return;
                Random rng = new Random();
                int n = shuffleIndices.Length;
                while (n > 1)
                {
                    n--;
                    int k = rng.Next(n + 1);
                    int temp = shuffleIndices[k];
                    shuffleIndices[k] = shuffleIndices[n];
                    shuffleIndices[n] = temp;
                }
            }

            // Helper method to serialize a float array to a byte array
            byte[] SerializeFloatArray(float[] data)
            {
                if (data == null) return new byte[0];
                var byteList = new List<byte>();
                foreach (var f in data) byteList.AddRange(BitConverter.GetBytes(f));
                return byteList.ToArray();
            }

            // Helper method to deserialize a byte array back to a float array
            float[] DeserializeFloatArray(byte[] data)
            {
                if (data == null || data.Length == 0) return new float[0]; // Handle empty array case
                if (data.Length % 4 != 0) // Size of float is 4 bytes
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: Byte array length ({data.Length}) is not a multiple of 4 for deserialization. Returning empty array.");
                    return new float[0]; // Return empty array on invalid size
                }
                var floatArray = new float[data.Length / 4];
                System.Buffer.BlockCopy(data, 0, floatArray, 0, data.Length);
                return floatArray;
            }

            #endregion


            try
            {
                // ------------------------------------------
                // Find or Create Core Outcome Record and Dependencies
                // ------------------------------------------
                retrievedOrNewOutcomeRecord = InMemoryTestDataSet.SimulatedCoreOutcomes
                                        .FirstOrDefault(r => r.AssociatedCustomerIdentifier == customerIdentifier);

                isNewRecord = (retrievedOrNewOutcomeRecord == null); // isNewRecord is now in scope

                if (isNewRecord)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: No existing CoreMlOutcomeRecord found for Customer Identifier {customerIdentifier}. Initializing new record and associated dependencies.");

                    var nextAvailableRecordIdentifier = InMemoryTestDataSet.SimulatedCoreOutcomes
                                        .Count > 0 ? InMemoryTestDataSet.SimulatedCoreOutcomes.Max(r => r.RecordIdentifier) : 0;

                    retrievedOrNewOutcomeRecord = new CoreMlOutcomeRecord // Assign to the outer-scoped variable
                    {
                        AssociatedCustomerIdentifier = customerIdentifier,
                        OutcomeGenerationTimestamp = DateTime.UtcNow,
                        RecordIdentifier = nextAvailableRecordIdentifier + 1,
                        CategoricalClassificationIdentifier = null,
                        CategoricalClassificationDescription = null,
                        SerializedSimulatedModelData = new byte[0], // Initialize as empty, will be populated after training
                        AncillaryBinaryDataPayload = new byte[0],  // Initialize as empty
                        DerivedProductFeatureVector = null,
                        DerivedServiceBenefitVector = null
                    };

                    // Note: In a real scenario, adding to the list here might need locking if this unit could be called concurrently
                    // for the *same* customer ID, which isn't the case in this specific workflow design (sequential start).
                    // For simplicity in this example, assuming sequential access to InMemoryTestDataSet for a given customer ID within this unit.
                    lock (InMemoryTestDataSet.SimulatedCoreOutcomes)
                    {
                        InMemoryTestDataSet.SimulatedCoreOutcomes.Add(retrievedOrNewOutcomeRecord);
                    }

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Created new CoreMlOutcomeRecord with Identifier {retrievedOrNewOutcomeRecord.RecordIdentifier} for customer {customerIdentifier}");

                    // Simulate creation/lookup of associated dependency records - simplified
                    // For a real system, these would be fetched from a database or other service.
                    // Adding simple checks and add-if-not-exists logic for simulation.
                    var associatedCustomer = InMemoryTestDataSet.SimulatedCustomerContexts.FirstOrDefault(c => c.CustomerLinkIdentifier == customerIdentifier);
                    if (associatedCustomer == null) { Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Creating new AssociatedCustomerContext record for Customer {customerIdentifier}"); var maxId = InMemoryTestDataSet.SimulatedCustomerContexts.Count > 0 ? InMemoryTestDataSet.SimulatedCustomerContexts.Max(c => c.ContextIdentifier) : 0; associatedCustomer = new AssociatedCustomerContext { ContextIdentifier = maxId + 1, CustomerLinkIdentifier = customerIdentifier, CustomerPrimaryGivenName = $"Simulated FN {customerIdentifier}", CustomerFamilyName = $"Simulated LN {customerIdentifier}", CustomerContactPhoneNumber = $"555-cust-{customerIdentifier}", CustomerStreetAddress = $"123 Main St Sim {customerIdentifier}", AffiliatedCompanyName = $"Acme Inc. Sim {customerIdentifier}" }; lock (InMemoryTestDataSet.SimulatedCustomerContexts) InMemoryTestDataSet.SimulatedCustomerContexts.Add(associatedCustomer); Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Created AssociatedCustomerContext record with Identifier {associatedCustomer.ContextIdentifier}"); } else { Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Existing AssociatedCustomerContext record found for Customer {customerIdentifier}"); }

                    var associatedWorkOrder = InMemoryTestDataSet.SimulatedWorkOrders.FirstOrDefault(o => o.CustomerLinkIdentifier == customerIdentifier);
                    if (associatedWorkOrder == null) { Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Creating new OperationalWorkOrderRecord for Customer {customerIdentifier}"); var maxId = InMemoryTestDataSet.SimulatedWorkOrders.Count > 0 ? InMemoryTestDataSet.SimulatedWorkOrders.Max(o => o.OrderRecordIdentifier) : 0; associatedWorkOrder = new OperationalWorkOrderRecord { OrderRecordIdentifier = maxId + 1, CustomerLinkIdentifier = customerIdentifier, SpecificOrderIdentifier = customerIdentifier + 9000 }; lock (InMemoryTestDataSet.SimulatedWorkOrders) InMemoryTestDataSet.SimulatedWorkOrders.Add(associatedWorkOrder); Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Created OperationalWorkOrderRecord with Identifier {associatedWorkOrder.OrderRecordIdentifier}"); } else { Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Existing OperationalWorkOrderRecord found for Customer {customerIdentifier}"); }

                    var operationalEventRecord = InMemoryTestDataSet.SimulatedOperationalEvents.FirstOrDefault(e => e.CustomerLinkIdentifier == customerIdentifier);
                    if (operationalEventRecord == null) { Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Creating new MlInitialOperationEvent record for Customer {customerIdentifier}"); var maxId = InMemoryTestDataSet.SimulatedOperationalEvents.Count > 0 ? InMemoryTestDataSet.SimulatedOperationalEvents.Max(e => e.EventIdentifier) : 0; operationalEventRecord = new MlInitialOperationEvent { EventIdentifier = maxId + 1, CustomerLinkIdentifier = customerIdentifier, RelatedOrderIdentifier = customerIdentifier + 9000, InternalOperationIdentifier = customerIdentifier + 8000, EventPayloadData = new byte[] { (byte)customerIdentifier, 0xAA } }; lock (InMemoryTestDataSet.SimulatedOperationalEvents) InMemoryTestDataSet.SimulatedOperationalEvents.Add(operationalEventRecord); Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Created MlInitialOperationEvent record with Identifier {operationalEventRecord.EventIdentifier}"); } else { Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Existing MlInitialOperationEvent record found for Customer {customerIdentifier}"); }

                    var validationRecord = InMemoryTestDataSet.SimulatedOutcomeValidations.FirstOrDefault(v => v.CustomerLinkIdentifier == customerIdentifier);
                    if (validationRecord == null) { Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Creating new MlOutcomeValidationRecord for Customer {customerIdentifier}"); var maxId = InMemoryTestDataSet.SimulatedOutcomeValidations.Count > 0 ? InMemoryTestDataSet.SimulatedOutcomeValidations.Max(v => v.ValidationRecordIdentifier) : 0; validationRecord = new MlOutcomeValidationRecord { ValidationRecordIdentifier = maxId + 1, CustomerLinkIdentifier = customerIdentifier, RelatedOrderIdentifier = customerIdentifier + 9000, ValidationResultData = new byte[] { (byte)customerIdentifier, 0xBB } }; lock (InMemoryTestDataSet.SimulatedOutcomeValidations) InMemoryTestDataSet.SimulatedOutcomeValidations.Add(validationRecord); Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Created MlOutcomeValidationRecord record with Identifier {validationRecord.ValidationRecordIdentifier}"); } else { Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Existing MlOutcomeValidationRecord record found for Customer {customerIdentifier}"); }

                    var initialStageDataRecord = InMemoryTestDataSet.SimulatedInitialOperationalStages.FirstOrDefault(s => s.CustomerLinkIdentifier == customerIdentifier);
                    if (initialStageDataRecord == null) { Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Creating new InitialOperationalStageData record for Customer {customerIdentifier}"); var maxId = InMemoryTestDataSet.SimulatedInitialOperationalStages.Count > 0 ? InMemoryTestDataSet.SimulatedInitialOperationalStages.Max(s => s.StageIdentifier) : 0; initialStageDataRecord = new InitialOperationalStageData { StageIdentifier = maxId + 1, CustomerLinkIdentifier = customerIdentifier, RelatedOrderIdentifier = customerIdentifier + 9000, InternalOperationIdentifier = customerIdentifier + 8000, ProcessOperationalIdentifier = customerIdentifier + 7000, CustomerServiceOperationIdentifier = customerIdentifier + 6000, SalesProcessIdentifier = customerIdentifier + 5000, LinkedSubServiceA = 1, LinkedSubServiceB = 2, LinkedSubServiceC = 3, LinkedSubProductA = 1, LinkedSubProductB = 2, LinkedSubProductC = 3, StageSpecificData = $"Simulated Stage Data for Customer {customerIdentifier}", StageProductVectorSnapshot = $"Stage1_P_Simulated:{customerIdentifier}", StageServiceVectorSnapshot = $"Stage1_S_Simulated:{customerIdentifier}" }; lock (InMemoryTestDataSet.SimulatedInitialOperationalStages) InMemoryTestDataSet.SimulatedInitialOperationalStages.Add(initialStageDataRecord); Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Created InitialOperationalStageData record with Identifier {initialStageDataRecord.StageIdentifier}"); } else { Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Existing InitialOperationalStageData record found for Customer {customerIdentifier}"); }


                    // Store references to newly created/found records in RuntimeContext
                    RuntimeProcessingContext.StoreContextValue("AssociatedCustomerContextRecord", associatedCustomer);
                    RuntimeProcessingContext.StoreContextValue("OperationalWorkOrderRecord", associatedWorkOrder);
                    RuntimeProcessingContext.StoreContextValue("MlInitialOperationEventRecord", operationalEventRecord);
                    RuntimeProcessingContext.StoreContextValue("MlOutcomeValidationRecord", validationRecord);
                    RuntimeProcessingContext.StoreContextValue("InitialOperationalStageDataRecord", initialStageDataRecord);
                    RuntimeProcessingContext.StoreContextValue("CurrentCoreOutcomeRecord", retrievedOrNewOutcomeRecord);
                    RuntimeProcessingContext.StoreContextValue("CurrentCustomerIdentifier", customerIdentifier);

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Verification (RuntimeContext) - AssociatedCustomerContext Identifier: {associatedCustomer?.ContextIdentifier}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Verification (RuntimeContext) - OperationalWorkOrderRecord Identifier: {associatedWorkOrder?.OrderRecordIdentifier}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Verification (RuntimeContext) - MlInitialOperationEventRecord Identifier: {operationalEventRecord?.EventIdentifier}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Verification (RuntimeContext) - MlOutcomeValidationRecord Identifier: {validationRecord?.ValidationRecordIdentifier}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Verification (RuntimeContext) - InitialOperationalStageDataRecord Identifier: {initialStageDataRecord?.StageIdentifier}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Verification (RuntimeContext) - CurrentCoreOutcomeRecord Identifier: {retrievedOrNewOutcomeRecord?.RecordIdentifier}");
                }
                else
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Existing CoreMlOutcomeRecord found for Customer Identifier {customerIdentifier}. Proceeding with existing record {retrievedOrNewOutcomeRecord!.RecordIdentifier}.");
                    // Store reference to the existing record in RuntimeContext if not already done
                    if (RuntimeProcessingContext.RetrieveContextValue("CurrentCoreOutcomeRecord") == null)
                    {
                        RuntimeProcessingContext.StoreContextValue("CurrentCoreOutcomeRecord", retrievedOrNewOutcomeRecord);
                        RuntimeProcessingContext.StoreContextValue("CurrentCustomerIdentifier", customerIdentifier);
                    }
                }


                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Starting Actual Model C Training/Inference with combined numerical and word data.");

                // ------------------------------------------
                // Prepare combined input data (Numerical + Word Embeddings) for Model C
                // ------------------------------------------
                // Define sample numerical data - using fixed samples mirroring Step 7 data
                float[][] numericalSamples = new float[][] {
                          new float[] { 0.3f, 0.7f, 0.1f, 0.85f },
                          new float[] { 0.5f, 0.2f, 0.9f, 0.35f },
                          new float[] { 0.8f, 0.6f, 0.4f, 0.55f },
                          new float[] { 0.1f, 0.8f, 0.6f, 0.25f },
                          new float[] { 0.7f, 0.3f, 0.2f, 0.95f },
                          new float[] { 0.4f, 0.5f, 0.7f, 0.65f },
                          new float[] { 0.2f, 0.9f, 0.3f, 0.15f },
                          new float[] { 0.6f, 0.1f, 0.8f, 0.75f },
                          new float[] { 0.35f, 0.65f, 0.15f, 0.80f },
                          new float[] { 0.55f, 0.25f, 0.85f, 0.30f },
                          new float[] { 0.75f, 0.55f, 0.45f, 0.60f },
                          new float[] { 0.15f, 0.75f, 0.55f, 0.20f },
                          new float[] { 0.65f, 0.35f, 0.25f, 0.90f },
                          new float[] { 0.45f, 0.45f, 0.65f, 0.70f },
                          new float[] { 0.25f, 0.85f, 0.35f, 0.10f },
                          new float[] { 0.50f, 0.15f, 0.75f, 0.80f }
                      };

                // Define sample word samples - using fixed samples mirroring Step 7 data
                string[] wordSamples = new string[] {
      "market growth potential high", "customer satisfaction excellent", "product quality superior",
      "service delivery timely", "price competitiveness average", "brand recognition strong",
      "operational efficiency optimal", "supply chain resilient", "market segment expanding",
      "customer retention excellent", "product innovation substantial", "service response immediate",
      "price positioning competitive", "brand loyalty increasing", "operational costs decreasing",
      "supply reliability consistent"
  };

                // Ensure the number of samples matches
                int sampleCount = Math.Min(numericalSamples.Length, wordSamples.Length);
                bool skipTrainingFlag = false;

                if (sampleCount == 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: No valid training data generated for Model C. Skipping training.");
                    if (retrievedOrNewOutcomeRecord != null) // Ensure record exists before updating it
                    {
                        retrievedOrNewOutcomeRecord.SerializedSimulatedModelData = new byte[0]; // Store empty array to indicate no data
                        retrievedOrNewOutcomeRecord.AncillaryBinaryDataPayload = new byte[0];  // Store empty array
                    }
                    skipTrainingFlag = true; // Set flag to skip training block
                }


                if (!skipTrainingFlag)
                {
                    // Transform word samples into embeddings using the helper method
                    float[][] wordEmbeddings = TransformWordsToEmbeddings(wordSamples.Take(sampleCount).ToArray());

                    // Convert jagged arrays to multidimensional arrays for TensorFlow
                    float[,] numericalData = ConvertJaggedToMultidimensional(numericalSamples.Take(sampleCount).ToArray());
                    float[,] wordData = ConvertJaggedToMultidimensional(wordEmbeddings);

                    int numericalFeatureCount = numericalData.GetLength(1);
                    int wordFeatureCount = wordData.GetLength(1);
                    int totalInputFeatureCount = numericalFeatureCount + wordFeatureCount;

                    // Define simple target values based on combined numerical data (matching Step 7 target logic)
                    float[,] targetValues = new float[sampleCount, 1];
                    for (int i = 0; i < sampleCount; i++)
                    {
                        if (numericalData.GetLength(1) < 4) // Check column length safety
                        {
                            targetValues[i, 0] = 0.0f; // Default target if row is invalid
                            continue;
                        }
                        float x = numericalData[i, 0];
                        float y = numericalData[i, 1];
                        float z = numericalData[i, 2];
                        float p = numericalData[i, 3];

                        // Use a more complex formula that includes non-linear terms (matching Step 7 target logic)
                        targetValues[i, 0] = x * (float)Math.Cos(p) +
                                           y * (float)Math.Sin(p) +
                                           z * (float)Math.Cos(p / 2f) +
                                           x * y * z * 0.1f;
                    }

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Created {sampleCount} combined numerical and word samples for Model C training.");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Numerical features: {numericalFeatureCount}, Word embedding features: {wordFeatureCount}. Total input features: {totalInputFeatureCount}");


                    // ------------------------------------------
                    // TensorFlow Graph Definition and Training (Model C)
                    // ------------------------------------------
                    graph = tf.Graph();
                    graph.as_default();
                    {
                        mlSession = tf.Session(graph); // Create session with the new graph

                        // Define operations using the created graph and session
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Step 4 - Initializing Model C Architecture.");

                        // Define placeholders for numerical and word inputs
                        Tensor numericalInputPlaceholder = tf.placeholder(tf.float32, shape: (-1, numericalFeatureCount), name: "numerical_input_C");
                        Tensor wordInputPlaceholder = tf.placeholder(tf.float32, shape: (-1, wordFeatureCount), name: "word_input_C");
                        Tensor targetOutputPlaceholder = tf.placeholder(tf.float32, shape: (-1, 1), name: "target_output_C");

                        // Concatenate the numerical and word input placeholders
                        var combinedInput = tf.concat(new[] { numericalInputPlaceholder, wordInputPlaceholder }, axis: 1, name: "combined_input_C");

                        // Try to load existing model parameters from the outcome record
                        // Use the outer-scoped retrievedOrNewOutcomeRecord
                        var existingWeights = retrievedOrNewOutcomeRecord?.SerializedSimulatedModelData == null || retrievedOrNewOutcomeRecord.SerializedSimulatedModelData.Length == 0 ?
                            null : DeserializeFloatArray(retrievedOrNewOutcomeRecord.SerializedSimulatedModelData);
                        var existingBias = retrievedOrNewOutcomeRecord?.AncillaryBinaryDataPayload == null || retrievedOrNewOutcomeRecord.AncillaryBinaryDataPayload.Length == 0 ?
                            null : DeserializeFloatArray(retrievedOrNewOutcomeRecord.AncillaryBinaryDataPayload);

                        int hiddenLayerSize = 64; // Default hidden size for Model C

                        int expectedTotalWeightCount = (totalInputFeatureCount * hiddenLayerSize) + (hiddenLayerSize * 1);
                        int expectedTotalBiasCount = hiddenLayerSize + 1;

                        bool useExistingParams = existingWeights != null && existingBias != null &&
                                                 existingWeights.Length == expectedTotalWeightCount && existingBias.Length == expectedTotalBiasCount;

                        ResourceVariable weights1, weights2, bias1, bias2;

                        if (!useExistingParams)
                        {
                            weights1 = tf.Variable(tf.random.normal((totalInputFeatureCount, hiddenLayerSize), stddev: 0.1f), name: "weights1_C");
                            bias1 = tf.Variable(tf.zeros(hiddenLayerSize, dtype: tf.float32), name: "bias1_C");
                            weights2 = tf.Variable(tf.random.normal((hiddenLayerSize, 1), stddev: 0.1f), name: "weights2_C");
                            bias2 = tf.Variable(tf.zeros(1, dtype: tf.float32), name: "bias2_C");
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Model C - Initializing NEW model parameters for combined input ({totalInputFeatureCount} -> {hiddenLayerSize} -> 1).");
                        }
                        else
                        {
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Model C - Loading EXISTING model parameters for combined input ({totalInputFeatureCount} -> {hiddenLayerSize} -> 1).");
                            float[] loadedWeights1 = new float[totalInputFeatureCount * hiddenLayerSize];
                            float[] loadedBias1 = new float[hiddenLayerSize];
                            float[] loadedWeights2 = new float[hiddenLayerSize * 1];
                            float[] loadedBias2 = new float[1];

                            int weightParamCountW1 = totalInputFeatureCount * hiddenLayerSize;
                            System.Buffer.BlockCopy(existingWeights!, 0, loadedWeights1, 0, loadedWeights1.Length * sizeof(float));
                            System.Buffer.BlockCopy(existingWeights!, weightParamCountW1 * sizeof(float), loadedWeights2, 0, loadedWeights2.Length * sizeof(float));

                            int biasParamCountB1 = hiddenLayerSize;
                            System.Buffer.BlockCopy(existingBias!, 0, loadedBias1, 0, loadedBias1.Length * sizeof(float));
                            System.Buffer.BlockCopy(existingBias!, biasParamCountB1 * sizeof(float), loadedBias2, 0, loadedBias2.Length * sizeof(float));

                            weights1 = tf.Variable(tf.constant(loadedWeights1.reshape(totalInputFeatureCount, hiddenLayerSize), dtype: tf.float32), name: "weights1_C");
                            bias1 = tf.Variable(tf.constant(loadedBias1.reshape(hiddenLayerSize), dtype: tf.float32), name: "bias1_C");
                            weights2 = tf.Variable(tf.constant(loadedWeights2.reshape(hiddenLayerSize, 1), dtype: tf.float32), name: "weights2_C");
                            bias2 = tf.Variable(tf.constant(loadedBias2.reshape(1), dtype: tf.float32), name: "bias2_C");
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Model C - Successfully loaded parameters.");
                        }

                        var hidden = tf.nn.relu(tf.add(tf.matmul(combinedInput, weights1), bias1), name: "hidden_C");
                        var predictions = tf.add(tf.matmul(hidden, weights2), bias2, name: "predictions_C");
                        Tensor lossOp = tf.reduce_mean(tf.square(tf.subtract(predictions, targetOutputPlaceholder)), name: "loss_C");
                        var optimizer = tf.train.AdamOptimizer(0.001f);
                        Operation trainOp = optimizer.minimize(lossOp);
                        Operation initOp = tf.global_variables_initializer();

                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] TensorFlow operations defined within Model C graph.");
                        mlSession.run(initOp);
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Model C - Actual TensorFlow.NET variables initialized.");

                        int numEpochs = 50;
                        int batchSize = 4;
                        int numBatches = (batchSize > 0 && sampleCount > 0) ? (int)Math.Ceiling((double)sampleCount / batchSize) : 0;
                        int[] indices = Enumerable.Range(0, sampleCount).ToArray();

                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Model C - Starting Actual Training Loop for {numEpochs} epochs with {numBatches} batches.");
                        for (int epoch = 0; epoch < numEpochs; epoch++)
                        {
                            ShuffleArray(indices);
                            float epochLoss = 0.0f;
                            for (int batch = 0; batch < numBatches; batch++)
                            {
                                int startIdx = batch * batchSize;
                                int endIdx = Math.Min(startIdx + batchSize, sampleCount);
                                int batchCount = endIdx - startIdx;
                                if (batchCount <= 0) continue;
                                float[,] batchNumerical = ExtractBatch(numericalData, indices, startIdx, batchCount);
                                float[,] batchWord = ExtractBatch(wordData, indices, startIdx, batchCount);
                                float[,] batchTarget = ExtractBatch(targetValues, indices, startIdx, batchCount);
                                var feeds = new FeedItem[] {
                                   new FeedItem(numericalInputPlaceholder, batchNumerical),
                                   new FeedItem(wordInputPlaceholder, batchWord),
                                   new FeedItem(targetOutputPlaceholder, batchTarget)
                               };
                                var fetches = new ITensorOrOperation[] { lossOp, trainOp };
                                var results = mlSession.run(fetches, feeds);
                                float currentBatchLoss = (float)((Tensor)results[0]).numpy()[0];
                                epochLoss += currentBatchLoss;
                                if (batch % 10 == 0 || batch == numBatches - 1)
                                {
                                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Epoch {epoch + 1}/{numEpochs}, Batch {batch + 1}/{numBatches}, Actual Batch Loss: {currentBatchLoss:E4}");
                                }
                            }
                            if (numBatches > 0) epochLoss /= numBatches; else epochLoss = float.NaN;
                            if (epoch % 10 == 0 || epoch == numEpochs - 1)
                            {
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Epoch {epoch + 1}/{numEpochs}, Average Epoch Loss: {(float.IsNaN(epochLoss) ? "N/A" : epochLoss.ToString("E4"))}");
                            }
                        }
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Model C training completed.");

                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Starting Actual Model C parameter serialization.");
                        var finalWeightsAndBias = mlSession.run(new[] { weights1, bias1, weights2, bias2 });
                        var finalWeights1Tensor = finalWeightsAndBias[0] as Tensor;
                        var finalBias1Tensor = finalWeightsAndBias[1] as Tensor;
                        var finalWeights2Tensor = finalWeightsAndBias[2] as Tensor;
                        var finalBias2Tensor = finalWeightsAndBias[3] as Tensor;

                        if (finalWeights1Tensor != null && finalBias1Tensor != null && finalWeights2Tensor != null && finalBias2Tensor != null)
                        {
                            var finalWeights1 = finalWeights1Tensor.ToArray<float>();
                            var finalBias1 = finalBias1Tensor.ToArray<float>();
                            var finalWeights2 = finalWeights2Tensor.ToArray<float>();
                            var finalBias2 = finalBias2Tensor.ToArray<float>();
                            var combinedWeights = finalWeights1.Concat(finalWeights2).ToArray();
                            var combinedBias = finalBias1.Concat(finalBias2).ToArray();
                            retrievedOrNewOutcomeRecord!.SerializedSimulatedModelData = SerializeFloatArray(combinedWeights);
                            retrievedOrNewOutcomeRecord.AncillaryBinaryDataPayload = SerializeFloatArray(combinedBias);
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Model C actual model parameters serialized to byte arrays (Weights size: {retrievedOrNewOutcomeRecord.SerializedSimulatedModelData.Length}, Bias size: {retrievedOrNewOutcomeRecord.AncillaryBinaryDataPayload.Length}).");
                        }
                        else
                        {
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Model C - Failed to fetch/serialize actual weights or bias tensors.");
                            retrievedOrNewOutcomeRecord!.SerializedSimulatedModelData = new byte[0];
                            retrievedOrNewOutcomeRecord.AncillaryBinaryDataPayload = new byte[0];
                        }
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Actual Model C parameter serialization completed.");
                    } // End of graph.as_default()
                } // End of !skipTrainingFlag


                // ------------------------------------------
                // Save Final State to Simulated Persistence and Runtime Context
                // ------------------------------------------
                if (retrievedOrNewOutcomeRecord != null)
                {
                    var recordIndex = InMemoryTestDataSet.SimulatedCoreOutcomes.FindIndex(r => r.RecordIdentifier == retrievedOrNewOutcomeRecord.RecordIdentifier);
                    if (recordIndex >= 0)
                    {
                        lock (InMemoryTestDataSet.SimulatedCoreOutcomes)
                        {
                            InMemoryTestDataSet.SimulatedCoreOutcomes[recordIndex] = retrievedOrNewOutcomeRecord;
                        }
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Model C actual parameter data saved successfully in simulated persistent storage.");
                    }
                    else
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Error - CoreMlOutcomeRecord with Identifier {retrievedOrNewOutcomeRecord.RecordIdentifier} not found in simulated storage during final update attempt!");
                    }
                    RuntimeProcessingContext.StoreContextValue("SequentialProcessingUnitC_SerializedModelData", retrievedOrNewOutcomeRecord.SerializedSimulatedModelData);
                    RuntimeProcessingContext.StoreContextValue("SequentialProcessingUnitC_AncillaryData", retrievedOrNewOutcomeRecord.AncillaryBinaryDataPayload);
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Model C actual model parameter data stored in Runtime Processing Context.");
                    var contextCustomerIdentifier = RuntimeProcessingContext.RetrieveContextValue("CurrentCustomerIdentifier");
                    var contextProcessOneData = RuntimeProcessingContext.RetrieveContextValue("SequentialProcessingUnitC_SerializedModelData") as byte[];
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Verification (RuntimeContext) - Customer Identifier: {contextCustomerIdentifier}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Verification (RuntimeContext) - Serialized Model Data Size: {contextProcessOneData?.Length ?? 0} bytes");
                }
                else
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: No outcome record to save to simulated persistence.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Unhandled Error in Sequential Initial Processing Unit C: {ex.Message}");
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Stack Trace: {ex.StackTrace}");
                if (retrievedOrNewOutcomeRecord != null)
                {
                    retrievedOrNewOutcomeRecord.SerializedSimulatedModelData = new byte[0];
                    retrievedOrNewOutcomeRecord.AncillaryBinaryDataPayload = new byte[0];
                    retrievedOrNewOutcomeRecord.CategoricalClassificationDescription = (retrievedOrNewOutcomeRecord.CategoricalClassificationDescription ?? "") + " (TrainingError)";
                    var recordIndex = InMemoryTestDataSet.SimulatedCoreOutcomes.FindIndex(r => r.RecordIdentifier == retrievedOrNewOutcomeRecord.RecordIdentifier);
                    if (recordIndex >= 0)
                    {
                        lock (InMemoryTestDataSet.SimulatedCoreOutcomes)
                        {
                            InMemoryTestDataSet.SimulatedCoreOutcomes[recordIndex] = retrievedOrNewOutcomeRecord;
                        }
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Updated simulated persistent storage with error state.");
                    }
                }
                else
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Error occurred but retrievedOrNewOutcomeRecord was null.");
                }
                throw;
            }
            finally
            {
                MlProcessOrchestrator.DisposeGraphAndSession(ref graph, ref mlSession); // Use the static helper
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Model C TF Graph and Session disposed.");
                RuntimeProcessingContext.StoreContextValue("SequentialProcessingUnitC_ActiveStatus", false);
                bool isActiveAfterExecution = (bool)RuntimeProcessingContext.RetrieveContextValue("SequentialProcessingUnitC_ActiveStatus")!;
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: SequentialProcessingUnitC ActiveStatus property value after execution: {isActiveAfterExecution}");
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Sequential Initial Processing Unit C (Actual Model C) finished.");
            }
        }
































        /// <summary>
        /// Processes data simulating Model A (ParallelProcessingUnitA).
        /// This method is designed to run in parallel with ParallelProcessingUnitB (Actual Model B).
        /// It orchestrates a multi-stage data processing and machine learning workflow,
        /// including data collection, feature engineering, tensor generation, quality checks,
        /// model training using TensorFlow.NET, and generating performance projections.
        /// </summary>
       /// <summary>
            /// This C# summary outlines key enhancements centered around a dynamic "Expression Proliferation" mechanism,
            /// integrated into an N-Dimensional embedding and neural network training process.
            ///
            /// Key Changes Implemented:
            ///
            /// 1.  Expression Proliferation:
            ///     The core mathematical expression evolved from a static "1+1" to a dynamic "1+P".
            ///     'P' represents the proliferation parameter, which starts at 1 and increments by 1
            ///     every 10 training epochs (P = 1 + floor(epoch / 10)).
            ///
            /// 2.  N-Dimensional Embedding Integration:
            ///     A new `ComputeNDimensionalEmbedding()` method was added. This method implements the
            ///     complete NDimensionalClusterExecution formula, which encompasses:
            ///     - Density-weighted clustering
            ///     - Geometric lifting
            ///     - Tensor field computation
            ///     - Velocity computations.
            ///     This embedding is calculated in an n-dimensional space for each instance of 'P'
            ///     (each proliferation state) during the training phase.
            ///
            /// 3.  Clear Sequential Process Documentation:
            ///     The enhanced process follows a clear sequence:
            ///     STEP 1→2: The "1+P" expression is processed (e.g., via Regex) and feeds into the N-Dimensional formula.
            ///     STEP 3: The N-dimensional embedding is computed using the current 'P' value via cluster execution.
            ///     STEP 4: Curvature is applied, with scaling influenced by the proliferation parameter 'P'.
            ///     STEP 5: Weight generation within the network is influenced by 'P'.
            ///     STEP 6: Vertex mask calculations also incorporate 'P'.
            ///     STEP 7: The training loop iteratively updates 'P' and recalculates affected components.
            ///     STEP 8: Performance projection now includes metrics reflecting the impact of proliferation.
            ///
            /// 4.  Cluster Input Data Organization:
            ///     Processing for cluster input data has been separated and clearly delineated with section headers.
            ///     This organization facilitates future dynamic generation of input data and includes
            ///     comprehensive logging of the clustering process.
            ///
            /// 5.  Proliferation During Training:
            ///     The proliferation mechanism is active throughout training:
            ///     - Each training epoch calculates the current proliferation instance: P = 1 + (epoch / 10).
            ///     - The N-dimensional embedding is recomputed for each training batch using the current 'P' value.
            ///     - The "1+P" expression, through the value of 'P' and the derived embedding, influences
            ///       vertex masks, curvature coefficients, and weight generation.
            ///     - The formula effectively proliferates through the network's architecture during training iterations.
            ///
            /// 6.  Complete Integration:
            ///     All stages of the model and training pipeline now cohesively incorporate the proliferation mechanism.
            ///     The TensorFlow architecture has been extended to accept the N-dimensional embedding as input.
            ///     Vertex masks are applied to hidden layers with scaling dependent on 'P'.
            ///     The final trained model includes metadata related to the proliferation process, including the final 'P' value.
            ///
            /// The method now demonstrates the complete process of how the "1+P" expression proliferates
            /// through the n-dimensional space during training, with each iteration computing the full
            /// cluster execution formula as an embedding that influences the neural network's learning process.
            /// </summary>
        /// <param name="outcomeRecord">The core CoreMlOutcomeRecord object being processed.</param>
        /// <param name="customerIdentifier">The identifier for the associated customer.</param>
        /// <param name="requestSequenceIdentifier">A unique identifier for the current workflow request session.</param>
        /// <param name="mlSession_param_unused">A dedicated actual TensorFlow.NET Session environment for ML tasks within this unit (now unused internally by this unit's TF ops).</param>
        /// <param name="unitResultsStore">A thread-safe dictionary to store results and state for subsequent processing units.</param>
        private async Task ParallelProcessingUnitA(CoreMlOutcomeRecord outcomeRecord, int customerIdentifier, int requestSequenceIdentifier, Tensorflow.Session mlSession_param_unused, ConcurrentDictionary<string, object> unitResultsStore)
        {
            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Starting Parallel Processing Unit A for customer {customerIdentifier}.");
            Tensorflow.Graph modelAGraph = null;
            Tensorflow.Session modelASession = null;

            try
            {
                //==========================================================================
                // General Utility Methods (Accessible by all Stages)
                //==========================================================================

                #region Helper Methods (Required by this method)

                /// <summary>
                /// STEP 1: EXPRESSION → REGEX CONVERSION WITH PROLIFERATION
                /// Converts a mathematical expression with proliferation parameter into a regular expression pattern.
                /// The proliferation parameter P allows the expression to scale across n-dimensional space.
                /// </summary>
                string ConvertExpressionToRegex(string expression)
                {
                    // Handle the proliferated expression "1+P" where P is the proliferation parameter
                    if (expression == "1+P")
                    {
                        return @"(\d+)([\+\-\*\/])(P)"; // Capture proliferation parameter
                    }

                    // Legacy support for "1+1" 
                    if (expression == "1+1")
                    {
                        return @"(\d+)([\+\-\*\/])(\d+)";
                    }

                    // For more complex expressions, implement a more sophisticated parser
                    string pattern = expression.Replace("1", @"(\d+)");
                    pattern = pattern.Replace("+", @"([\+\-\*\/])");
                    pattern = pattern.Replace("P", @"(P)"); // Handle proliferation parameter

                    return pattern;
                }

                /// <summary>
                /// STEP 2: REGEX → N-DIMENSIONAL EXPRESSION CONVERSION WITH PROLIFERATION
                /// Converts a regular expression pattern into an n-dimensional compute-safe expression.
                /// This is where the proliferation parameter P gets embedded into the dimensional formula.
                /// The resulting expression will be applied to vertices and modified by curvature during training.
                /// </summary>
                string ConvertRegexToNDimensionalExpression(string regexPattern, int proliferationInstance = 1)
                {
                    // Handle proliferated pattern "1+P" - the core of our dimensional expansion
                    if (regexPattern == @"(\d+)([\+\-\*\/])(P)" ||
                        (regexPattern?.Contains(@"(\d+)") == true && regexPattern?.Contains(@"(P)") == true))
                    {
                        // PROLIFERATION: P becomes the instance multiplier for dimensional scaling
                        // This formula will be applied to vertex masks and modified by curvature coefficients
                        return $"ND(x,y,z,p)=Vx*cos(p*{proliferationInstance})+Vy*sin(p*{proliferationInstance})+Vz*cos(p*{proliferationInstance}/2)";
                    }

                    // Legacy pattern support
                    if (regexPattern == @"(\d+)([\+\-\*\/])(\d+)" ||
                        (regexPattern?.Contains(@"(\d+)") == true && regexPattern?.Contains(@"([\+\-\*\/])") == true))
                    {
                        return "ND(x,y,z,p)=Vx*cos(p)+Vy*sin(p)+Vz*cos(p/2)";
                    }

                    // Default fallback for unrecognized patterns
                    return "ND(x,y,z,p)=x+y+z";
                }

                /// <summary>
                /// STEP 3: N-DIMENSIONAL EMBEDDING COMPUTATION WITH PROLIFERATION
                /// This embedding computes the full NDimensionalClusterExecution formula in n-dimensional space
                /// for each proliferation instance during training iterations.
                /// The embedding integrates density-weighted clustering, geometric lifting, tensor fields, and velocity computations.
                /// </summary>
                /// <param name="inputData">Current training batch data</param>
                /// <param name="proliferationInstance">Current proliferation instance (P value)</param>
                /// <param name="embeddingDimension">Target embedding dimension</param>
                /// <returns>N-dimensional embedding incorporating cluster execution results</returns>
                float[,] ComputeNDimensionalEmbedding(float[,] inputData, int proliferationInstance, int embeddingDimension = 16)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Computing N-Dimensional Embedding with proliferation instance: {proliferationInstance}");

                    if (inputData == null || inputData.GetLength(0) == 0)
                    {
                        return new float[1, embeddingDimension];
                    }

                    int batchSize = inputData.GetLength(0);
                    int inputFeatures = inputData.GetLength(1);

                    // Convert to double array for high-precision cluster computation
                    double[][] dataPoints = new double[batchSize][];
                    for (int i = 0; i < batchSize; i++)
                    {
                        dataPoints[i] = new double[inputFeatures];
                        for (int j = 0; j < inputFeatures; j++)
                        {
                            dataPoints[i][j] = inputData[i, j];
                        }
                    }

                    // === N-DIMENSIONAL CLUSTER EXECUTION EMBEDDING ===
                    // Compute the full NDimensionalClusterExecution formula as an embedding

                    // Dynamic parameters based on proliferation
                    int K = Math.Min(Math.Max(2, proliferationInstance + 1), batchSize); // Clusters scale with proliferation
                    double r = 0.3 + (proliferationInstance * 0.1); // Radius scales with proliferation
                    int maxIter = 10 + (proliferationInstance * 2); // Iterations scale with proliferation

                    // === PHASE 1: DATA PREPROCESSING WITH PROLIFERATION SCALING ===
                    double[] minValues = new double[inputFeatures];
                    double[] maxValues = new double[inputFeatures];

                    for (int j = 0; j < inputFeatures; j++)
                    {
                        minValues[j] = dataPoints.Min(x => x[j]);
                        maxValues[j] = dataPoints.Max(x => x[j]);
                    }

                    // Min-Max Normalization with proliferation scaling
                    double[][] normalizedData = new double[batchSize][];
                    for (int i = 0; i < batchSize; i++)
                    {
                        normalizedData[i] = new double[inputFeatures];
                        for (int j = 0; j < inputFeatures; j++)
                        {
                            double range = maxValues[j] - minValues[j];
                            if (range > 1e-10)
                            {
                                normalizedData[i][j] = ((dataPoints[i][j] - minValues[j]) / range) * proliferationInstance;
                            }
                            else
                            {
                                normalizedData[i][j] = 0.5 * proliferationInstance;
                            }
                        }
                    }

                    // Local Density Computation with proliferation-adjusted radius
                    double[] densities = new double[batchSize];
                    double adjustedRadius = r * Math.Sqrt(proliferationInstance);

                    for (int i = 0; i < batchSize; i++)
                    {
                        for (int j = 0; j < batchSize; j++)
                        {
                            if (i != j)
                            {
                                double distance = 0;
                                for (int k = 0; k < inputFeatures; k++)
                                {
                                    double diff = normalizedData[i][k] - normalizedData[j][k];
                                    distance += diff * diff;
                                }
                                distance = Math.Sqrt(distance);

                                if (distance <= adjustedRadius)
                                {
                                    densities[i] += 1.0;
                                }
                            }
                        }
                    }

                    // === PHASE 2: DENSITY-NORMALIZED K-MEANS ===
                    var densityIndexPairs = densities.Select((density, index) => new { Density = density, Index = index })
                                                   .OrderByDescending(x => x.Density)
                                                   .Take(K)
                                                   .ToArray();

                    double[][] centroids = new double[K][];
                    for (int j = 0; j < K; j++)
                    {
                        int seedIndex = densityIndexPairs[j].Index;
                        centroids[j] = (double[])normalizedData[seedIndex].Clone();
                    }

                    // K-Means iteration with proliferation influence
                    int[] assignments = new int[batchSize];
                    for (int iter = 0; iter < maxIter; iter++)
                    {
                        bool changed = false;

                        // Assignment step
                        for (int i = 0; i < batchSize; i++)
                        {
                            int bestCluster = 0;
                            double bestDistance = double.MaxValue;

                            for (int j = 0; j < K; j++)
                            {
                                double distance = 0;
                                for (int k = 0; k < inputFeatures; k++)
                                {
                                    double diff = normalizedData[i][k] - centroids[j][k];
                                    distance += diff * diff;
                                }

                                if (distance < bestDistance)
                                {
                                    bestDistance = distance;
                                    bestCluster = j;
                                }
                            }

                            if (assignments[i] != bestCluster)
                            {
                                assignments[i] = bestCluster;
                                changed = true;
                            }
                        }

                        if (!changed) break;

                        // Update step with proliferation weighting
                        for (int j = 0; j < K; j++)
                        {
                            var clusterPoints = normalizedData.Where((point, index) => assignments[index] == j).ToArray();
                            if (clusterPoints.Length > 0)
                            {
                                for (int k = 0; k < inputFeatures; k++)
                                {
                                    centroids[j][k] = clusterPoints.Average(point => point[k]) * proliferationInstance;
                                }
                            }
                        }
                    }

                    // === PHASE 3: GEOMETRIC LIFTING TO ℝ^(d+1) ===
                    // Relative Centroid
                    double[] relativeCenter = new double[inputFeatures];
                    for (int k = 0; k < inputFeatures; k++)
                    {
                        relativeCenter[k] = centroids.Average(centroid => centroid[k]);
                    }

                    // Simplex Construction with proliferation height scaling
                    double relativeMagnitude = Math.Sqrt(relativeCenter.Sum(x => x * x));
                    double apexHeight = relativeMagnitude * proliferationInstance;

                    // === PHASE 4: TENSOR & VELOCITY COMPUTATION ===
                    // Base Centroid calculation
                    double[] baseCentroid = new double[inputFeatures + 1];
                    for (int j = 0; j < K; j++)
                    {
                        for (int k = 0; k < inputFeatures; k++)
                        {
                            baseCentroid[k] += centroids[j][k] / K;
                        }
                    }
                    baseCentroid[inputFeatures] = 0; // z = 0 for base

                    // Apex with proliferation scaling
                    double[] apex = new double[inputFeatures + 1];
                    for (int k = 0; k < inputFeatures; k++)
                    {
                        apex[k] = relativeCenter[k];
                    }
                    apex[inputFeatures] = apexHeight;

                    // Direction vector
                    double[] direction = new double[inputFeatures + 1];
                    for (int k = 0; k <= inputFeatures; k++)
                    {
                        direction[k] = apex[k] - baseCentroid[k];
                    }

                    // Magnitude & Unit Direction
                    double magnitude = Math.Sqrt(direction.Sum(x => x * x));
                    double[] unitDirection = new double[inputFeatures + 1];
                    if (magnitude > 1e-10)
                    {
                        for (int k = 0; k <= inputFeatures; k++)
                        {
                            unitDirection[k] = direction[k] / magnitude;
                        }
                    }

                    // Velocity Field with proliferation
                    double[] velocity = new double[inputFeatures + 1];
                    for (int k = 0; k <= inputFeatures; k++)
                    {
                        velocity[k] = magnitude * unitDirection[k] * proliferationInstance;
                    }

                    // === PHASE 5: EMBEDDING GENERATION ===
                    // Generate the final n-dimensional embedding from cluster execution results
                    float[,] embedding = new float[batchSize, embeddingDimension];

                    for (int i = 0; i < batchSize; i++)
                    {
                        int clusterIdx = assignments[i];
                        double density = densities[i];

                        for (int j = 0; j < embeddingDimension; j++)
                        {
                            // Combine multiple embedding components with proliferation influence
                            double component = 0;

                            // Component 1: Cluster centroid influence
                            if (j < inputFeatures && clusterIdx < centroids.Length)
                            {
                                component += centroids[clusterIdx][j % inputFeatures] * 0.3;
                            }

                            // Component 2: Velocity field influence
                            if (j < velocity.Length)
                            {
                                component += velocity[j] * 0.25;
                            }

                            // Component 3: Unit direction influence
                            if (j < unitDirection.Length)
                            {
                                component += unitDirection[j] * 0.25;
                            }

                            // Component 4: Density influence with proliferation
                            component += (density / batchSize) * proliferationInstance * 0.2;

                            // Apply trigonometric transformation based on n-dimensional formula
                            double p = (double)proliferationInstance;
                            double x = normalizedData[i][j % inputFeatures];
                            component = x * Math.Cos(p * j) + component * Math.Sin(p * j) + density * Math.Cos(p * j / 2);

                            embedding[i, j] = (float)component;
                        }
                    }

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Generated N-Dimensional embedding: {batchSize}x{embeddingDimension} with proliferation {proliferationInstance}");
                    return embedding;
                }

                /// <summary>
                /// Transforms word-based samples into numerical embeddings using a simplified embedding technique.
                /// </summary>
                float[][] TransformWordsToEmbeddings(string[] wordSamples)
                {
                    if (wordSamples == null) return new float[0][];

                    int embeddingDimensions = 10;
                    float[][] embeddings = new float[wordSamples.Length][];

                    for (int i = 0; i < wordSamples.Length; i++)
                    {
                        embeddings[i] = new float[embeddingDimensions];
                        string[] words = wordSamples[i]?.Split(' ') ?? new string[0];

                        for (int j = 0; j < words.Length; j++)
                        {
                            string word = words[j];
                            if (string.IsNullOrEmpty(word)) continue;

                            int hashBase = word.GetHashCode();
                            for (int k = 0; k < embeddingDimensions; k++)
                            {
                                int valueInt = Math.Abs(hashBase * (k + 1) * (j + 1) * 31);
                                float value = (valueInt % 1000) / 1000.0f;
                                embeddings[i][k] += value * (1.0f / (j + 1.0f));
                            }
                        }

                        // Normalize the embedding vector
                        float magnitudeSq = 0;
                        for (int k = 0; k < embeddingDimensions; k++)
                        {
                            magnitudeSq += embeddings[i][k] * embeddings[i][k];
                        }

                        float magnitude = (float)Math.Sqrt(magnitudeSq);
                        if (magnitude > 1e-6f)
                        {
                            for (int k = 0; k < embeddingDimensions; k++)
                            {
                                embeddings[i][k] /= magnitude;
                            }
                        }
                    }

                    return embeddings;
                }

                /// <summary>
                /// STEP 4: N-DIMENSIONAL EXPRESSION → CURVATURE APPLICATION WITH PROLIFERATION
                /// Applies the n-dimensional expression to curvature coefficients.
                /// This is where the proliferated formula begins to influence the geometric properties.
                /// </summary>
                float[] ApplyNDimensionalExpressionToCurvature(float[] coefficients, string ndExpression, int proliferationInstance = 1)
                {
                    if (coefficients == null) return new float[0];

                    float[] modifiedCoefficients = new float[coefficients.Length];
                    System.Buffer.BlockCopy(coefficients, 0, modifiedCoefficients, 0, coefficients.Length * sizeof(float));

                    // Apply the proliferated expression's effect to the coefficients
                    if (ndExpression.StartsWith("ND(x,y,z,p)="))
                    {
                        // PROLIFERATION EFFECT: The proliferation instance scales the amplification
                        // This amplification will affect how the expression influences vertex weights during training
                        float baseAmplification = 2.0f; // Base "1+1" effect
                        float proliferationAmplification = baseAmplification * proliferationInstance; // Scaled by P

                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Applying proliferation amplification factor: {proliferationAmplification} (base: {baseAmplification} * instance: {proliferationInstance})");

                        // Apply to primary diagonal coefficients (xx, yy, zz) - these represent the main dimensional influences
                        if (modifiedCoefficients.Length > 0) modifiedCoefficients[0] *= proliferationAmplification;  // xx coefficient
                        if (modifiedCoefficients.Length > 1) modifiedCoefficients[1] *= proliferationAmplification;  // yy coefficient
                        if (modifiedCoefficients.Length > 2) modifiedCoefficients[2] *= proliferationAmplification;  // zz coefficient

                        // Scale cross-terms relative to the proliferated amplification
                        float crossTermScale = proliferationAmplification * 0.75f; // Slightly less influence for cross-terms
                        if (modifiedCoefficients.Length > 3) modifiedCoefficients[3] *= crossTermScale;  // xy coefficient
                        if (modifiedCoefficients.Length > 4) modifiedCoefficients[4] *= crossTermScale;  // xz coefficient
                        if (modifiedCoefficients.Length > 5) modifiedCoefficients[5] *= crossTermScale;  // yz coefficient

                        // Apply graduated influence to higher-order terms
                        float higherOrderScale = (crossTermScale + 1.0f) / 2.0f;
                        if (modifiedCoefficients.Length > 6) modifiedCoefficients[6] *= higherOrderScale;
                        if (modifiedCoefficients.Length > 7) modifiedCoefficients[7] *= higherOrderScale;
                        if (modifiedCoefficients.Length > 8) modifiedCoefficients[8] *= higherOrderScale;
                    }

                    return modifiedCoefficients;
                }

                /// <summary>
                /// STEP 5: EXPRESSION → WEIGHT GENERATION WITH PROLIFERATION
                /// Generates weight matrices from our proliferated n-dimensional expression.
                /// These weights will be applied to vertices and modified during training iterations.
                /// </summary>
                float[,] GenerateWeightsFromExpression(string expression, int inputDim, int outputDim, int proliferationInstance = 1)
                {
                    if (inputDim <= 0 || outputDim <= 0)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper: GenerateWeightsFromExpression received invalid dimensions ({inputDim}, {outputDim}). Returning empty array.");
                        return new float[0, 0];
                    }

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Generating weights with proliferation instance: {proliferationInstance}");

                    float[,] weights = new float[inputDim, outputDim];
                    Random rand = new Random(42 + inputDim * 100 + outputDim + proliferationInstance * 17); // Include proliferation in seed

                    for (int i = 0; i < inputDim; i++)
                    {
                        for (int j = 0; j < outputDim; j++)
                        {
                            float baseWeight = (float)(rand.NextDouble() * 0.04 - 0.02);

                            // PROLIFERATION INFLUENCE: Scale trigonometric functions by proliferation instance
                            // This creates the oscillatory pattern that will be applied to vertices
                            float proliferationFactor = proliferationInstance * 0.5f; // Scale factor for proliferation

                            float expressionInfluence = (float)(
                                Math.Cos((i + j) * Math.PI / (inputDim + outputDim) * proliferationFactor) +
                                Math.Sin(i * Math.PI / inputDim * proliferationFactor) * 0.5 +
                                Math.Cos(j * Math.PI / (outputDim * 2.0) * proliferationFactor) * 0.5
                            );

                            float influenceScale = 0.1f * proliferationInstance; // Scale influence by proliferation
                            weights[i, j] = baseWeight + expressionInfluence * influenceScale;
                        }
                    }

                    // VERTEX ENHANCEMENT: Boost corner weights (outermost vertices) based on proliferation
                    float cornerBoost = 1.5f + (proliferationInstance - 1) * 0.2f; // Increase boost with proliferation
                    if (inputDim > 0 && outputDim > 0)
                    {
                        weights[0, 0] *= cornerBoost;
                        weights[0, outputDim - 1] *= cornerBoost;
                        weights[inputDim - 1, 0] *= cornerBoost;
                        weights[inputDim - 1, outputDim - 1] *= cornerBoost;
                    }

                    return weights;
                }

                /// <summary>
                /// Calculates a basis vector from sample coordinates along a specific dimension.
                /// </summary>
                System.Numerics.Vector3 CalculateBasisVector(System.Numerics.Vector3[] coordinates, int dimension)
                {
                    if (coordinates == null || coordinates.Length == 0 || dimension < 0 || dimension > 2)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: Invalid input for CalculateBasisVector. Returning zero vector.");
                        return System.Numerics.Vector3.Zero;
                    }

                    System.Numerics.Vector3 basis = System.Numerics.Vector3.Zero;

                    foreach (var coord in coordinates)
                    {
                        float component = dimension == 0 ? coord.X : (dimension == 1 ? coord.Y : coord.Z);
                        basis += new System.Numerics.Vector3(
                            coord.X * component,
                            coord.Y * component,
                            coord.Z * component
                        );
                    }

                    float magnitude = basis.Length();
                    if (magnitude > 1e-6f)
                    {
                        basis = System.Numerics.Vector3.Divide(basis, magnitude);
                    }

                    return basis;
                }

                /// <summary>
                /// Calculates coefficients that represent how the curvature varies in the sample space.
                /// These coefficients will be modified by the proliferated expression.
                /// </summary>
                float[] CalculateCurvatureCoefficients(System.Numerics.Vector3[] coordinates, System.Numerics.Vector3[] values)
                {
                    if (coordinates == null || values == null || coordinates.Length == 0 || coordinates.Length != values.Length)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: Invalid input for CalculateCurvatureCoefficients. Returning zero coefficients.");
                        return new float[9];
                    }

                    float[] coefficients = new float[9];

                    for (int i = 0; i < coordinates.Length; i++)
                    {
                        System.Numerics.Vector3 coord = coordinates[i];
                        System.Numerics.Vector3 value = values[i];

                        float x2 = coord.X * coord.X;
                        float y2 = coord.Y * coord.Y;
                        float z2 = coord.Z * coord.Z;
                        float xy = coord.X * coord.Y;
                        float xz = coord.X * coord.Z;
                        float yz = coord.Y * coord.Z;

                        float dot = System.Numerics.Vector3.Dot(coord, value);

                        coefficients[0] += x2 * dot; // xx component
                        coefficients[1] += y2 * dot; // yy component
                        coefficients[2] += z2 * dot; // zz component
                        coefficients[3] += xy * dot; // xy component
                        coefficients[4] += xz * dot; // xz component
                        coefficients[5] += yz * dot; // yz component
                        coefficients[6] += x2 * y2 * dot; // xxyy component (higher order)
                        coefficients[7] += x2 * z2 * dot; // xxzz component (higher order)
                        coefficients[8] += y2 * z2 * dot; // yyzz component (higher order)
                    }

                    if (coordinates.Length > 0)
                    {
                        for (int i = 0; i < coefficients.Length; i++)
                        {
                            coefficients[i] /= coordinates.Length;
                        }
                    }

                    return coefficients;
                }

                /// <summary>
                /// Calculates the eigenvalues of the curvature tensor.
                /// </summary>
                float[] CalculateEigenvalues(float[] coefficients)
                {
                    if (coefficients == null || coefficients.Length < 6)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: Invalid input for CalculateEigenvalues. Returning default eigenvalues.");
                        return new float[] { 1.0f, 1.0f, 1.0f };
                    }

                    float[,] matrix = new float[3, 3];
                    matrix[0, 0] = coefficients[0]; // xx
                    matrix[1, 1] = coefficients[1]; // yy
                    matrix[2, 2] = coefficients[2]; // zz
                    matrix[0, 1] = matrix[1, 0] = coefficients[3]; // xy
                    matrix[0, 2] = matrix[2, 0] = coefficients[4]; // xz
                    matrix[1, 2] = matrix[2, 1] = coefficients[5]; // yz

                    float trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2];
                    float det = matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1]) -
                                matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0]) +
                                matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0]);

                    float[] eigenvalues = new float[3];
                    eigenvalues[0] = trace / 3.0f + 0.1f * det;
                    eigenvalues[1] = trace / 3.0f;
                    eigenvalues[2] = trace / 3.0f - 0.1f * det;

                    return eigenvalues;
                }

                /// <summary>
                /// Converts eigenvalues to weights for loss function.
                /// </summary>
                float[] ConvertEigenvaluesToWeights(float[] eigenvalues)
                {
                    if (eigenvalues == null || eigenvalues.Length == 0)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: Eigenvalues array is null or empty for weights. Returning default weights.");
                        return new float[] { 1.0f };
                    }

                    float[] weights = new float[eigenvalues.Length];
                    float sumAbsEigenvalues = 0.0f;

                    for (int i = 0; i < eigenvalues.Length; i++)
                    {
                        weights[i] = Math.Abs(eigenvalues[i]);
                        sumAbsEigenvalues += weights[i];
                    }

                    if (sumAbsEigenvalues > 1e-6f)
                    {
                        float maxAbsEigenvalue = weights.Max();
                        if (maxAbsEigenvalue > 1e-6f)
                        {
                            for (int i = 0; i < weights.Length; i++)
                            {
                                weights[i] = 0.5f + 0.5f * (weights[i] / maxAbsEigenvalue);
                            }
                        }
                        else
                        {
                            for (int i = 0; i < weights.Length; i++)
                            {
                                weights[i] = 1.0f;
                            }
                        }

                        return new float[] { weights.Average() };
                    }
                    else
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: Sum of absolute eigenvalues is zero. Returning default weights.");
                        return new float[] { 1.0f };
                    }
                }

                /// <summary>
                /// STEP 6: VERTEX MASK CALCULATION WITH PROLIFERATION
                /// Calculates a mask that identifies the outermost vertices in a tensor.
                /// This is where the proliferated expression gets applied to the vertex structure.
                /// The mask will be used during training to focus the proliferated formula on key vertices.
                /// </summary>
                Tensor CalculateOutermostVertexMask(Tensor input, int proliferationInstance = 1)
                {
                    if (input == null || input.shape.rank < 2)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper: CalculateOutermostVertexMask received null or insufficient rank tensor. Returning default mask.");
                        return tf.ones_like(input);
                    }

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Calculating vertex mask with proliferation instance: {proliferationInstance}");

                    var shape = tf.shape(input);
                    var batchSize = tf.slice(shape, begin: new int[] { 0 }, size: new int[] { 1 });
                    var features = tf.slice(shape, begin: new int[] { 1 }, size: new int[] { 1 });

                    var featureIndices = tf.cast(tf.range(0, features), dtype: tf.float32);
                    var normalizedIndices = tf.divide(featureIndices, tf.cast(features - 1, tf.float32));

                    // PROLIFERATION EFFECT ON VERTEX PATTERN: Scale the vertex emphasis by proliferation instance
                    float proliferationScale = 2.0f * proliferationInstance; // Base scale * proliferation
                    var featureMask = tf.multiply(tf.abs(normalizedIndices - 0.5f), proliferationScale, name: "proliferated_vertex_mask");

                    var batchSizeInt = tf.cast(batchSize, tf.int32);
                    var expandedMask = tf.tile(tf.reshape(featureMask, shape: new int[] { 1, -1 }),
                                             multiples: tf.concat(new[] { batchSizeInt, tf.constant(new int[] { 1 }) }, axis: 0),
                                             name: "expanded_proliferated_mask");

                    return expandedMask;
                }

                /// <summary>
                /// Converts a jagged array to a multidimensional array.
                /// </summary>
                float[,] ConvertJaggedToMultidimensional(float[][] jaggedArray)
                {
                    if (jaggedArray == null || jaggedArray.Length == 0 || jaggedArray.Any(row => row == null))
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper: ConvertJaggedToMultidimensional received null, empty, or jagged array with null rows. Returning empty multidimensional array.");
                        return new float[0, 0];
                    }
                    if (jaggedArray[0].Length == 0)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper: ConvertJaggedToMultidimensional received jagged array with zero columns. Returning empty multidimensional array.");
                        return new float[jaggedArray.Length, 0];
                    }

                    int rows = jaggedArray.Length;
                    int cols = jaggedArray[0].Length;
                    float[,] result = new float[rows, cols];

                    for (int i = 0; i < rows; i++)
                    {
                        if (jaggedArray[i].Length != cols)
                        {
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: Row {i} in jagged array has inconsistent length ({jaggedArray[i].Length} vs {cols}). Returning partial result.");
                            int currentCols = jaggedArray[i].Length;
                            for (int j = 0; j < Math.Min(cols, currentCols); j++)
                            {
                                result[i, j] = jaggedArray[i][j];
                            }
                        }
                        else
                        {
                            System.Buffer.BlockCopy(jaggedArray[i], 0, result, i * cols * sizeof(float), cols * sizeof(float));
                        }
                    }

                    return result;
                }

                /// <summary>
                /// Extracts a batch from a multidimensional array using indices.
                /// </summary>
                float[,] ExtractBatch(float[,] data, int[] batchIndices, int startIdx, int count)
                {
                    if (data == null || batchIndices == null || data.GetLength(0) == 0 || batchIndices.Length == 0)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: Invalid or empty input data/indices for ExtractBatch. Returning empty batch.");
                        return new float[0, data?.GetLength(1) ?? 0];
                    }
                    if (batchIndices.Length < startIdx + count || data.GetLength(0) < batchIndices.Max() + 1)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: Indices or data size mismatch for ExtractBatch (data rows: {data.GetLength(0)}, indices length: {batchIndices.Length}, startIdx: {startIdx}, count: {count}, max index: {batchIndices.Max()}). Returning empty batch.");
                        return new float[0, data.GetLength(1)];
                    }

                    if (count <= 0) return new float[0, data.GetLength(1)];

                    int cols = data.GetLength(1);
                    float[,] batch = new float[count, cols];

                    for (int i = 0; i < count; i++)
                    {
                        int srcIdx = batchIndices[startIdx + i];
                        if (srcIdx < 0 || srcIdx >= data.GetLength(0))
                        {
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Error: Invalid index {srcIdx} in ExtractBatch indices array at batch index {i}. Stopping batch extraction.");
                            var partialBatch = new float[i, cols];
                            System.Buffer.BlockCopy(batch, 0, partialBatch, 0, i * cols * sizeof(float));
                            return partialBatch;
                        }
                        System.Buffer.BlockCopy(data, srcIdx * cols * sizeof(float), batch, i * cols * sizeof(float), cols * sizeof(float));
                    }

                    return batch;
                }

                /// <summary>
                /// Shuffles an array randomly.
                /// </summary>
                void ShuffleArray(int[] shuffleIndices)
                {
                    if (shuffleIndices == null) return;
                    Random rng = new Random();
                    int n = shuffleIndices.Length;

                    while (n > 1)
                    {
                        n--;
                        int k = rng.Next(n + 1);
                        int temp = shuffleIndices[k];
                        shuffleIndices[k] = shuffleIndices[n];
                        shuffleIndices[n] = temp;
                    }
                }

                /// <summary>
                /// Serializes model metadata to JSON.
                /// </summary>
                string SerializeMetadata(Dictionary<string, object> metadata)
                {
                    if (metadata == null) return "{}";

                    StringBuilder sb = new StringBuilder();
                    sb.Append("{");

                    bool first = true;
                    foreach (var entry in metadata)
                    {
                        if (!first) sb.Append(",");

                        sb.Append($"\"{entry.Key}\":");

                        if (entry.Value is string strValue)
                        {
                            sb.Append($"\"{strValue.Replace("\"", "\\\"")}\"");
                        }
                        else if (entry.Value is float[] floatArray)
                        {
                            sb.Append("[");
                            sb.Append(string.Join(",", floatArray.Select(f => f.ToString("F6"))));
                            sb.Append("]");
                        }
                        else if (entry.Value is double[] doubleArray)
                        {
                            sb.Append("[");
                            sb.Append(string.Join(",", doubleArray.Select(d => d.ToString("F6"))));
                            sb.Append("]");
                        }
                        else if (entry.Value is int[] intArray)
                        {
                            sb.Append("[");
                            sb.Append(string.Join(",", intArray));
                            sb.Append("]");
                        }
                        else if (entry.Value is System.Numerics.Vector3 vector)
                        {
                            sb.Append($"[\"{vector.X:F6}\",\"{vector.Y:F6}\",\"{vector.Z:F6}\"]");
                        }
                        else if (entry.Value is float floatValue)
                        {
                            sb.Append(floatValue.ToString("F6"));
                        }
                        else if (entry.Value is double doubleValue)
                        {
                            sb.Append(doubleValue.ToString("F6"));
                        }
                        else if (entry.Value is int intValue)
                        {
                            sb.Append(intValue.ToString());
                        }
                        else if (entry.Value is bool boolValue)
                        {
                            sb.Append(boolValue.ToString().ToLower());
                        }
                        else if (entry.Value is float[,] float2DArray)
                        {
                            sb.Append("[");
                            for (int i = 0; i < float2DArray.GetLength(0); i++)
                            {
                                if (i > 0) sb.Append(",");
                                sb.Append("[");
                                for (int j = 0; j < float2DArray.GetLength(1); j++)
                                {
                                    if (j > 0) sb.Append(",");
                                    sb.Append(float2DArray[i, j].ToString("F6"));
                                }
                                sb.Append("]");
                            }
                            sb.Append("]");
                        }
                        else if (entry.Value != null)
                        {
                            try
                            {
                                string valueStr = entry.Value.ToString();
                                if (float.TryParse(valueStr, out _) || double.TryParse(valueStr, out _) || bool.TryParse(valueStr, out _))
                                {
                                    sb.Append(valueStr);
                                }
                                else
                                {
                                    sb.Append($"\"{valueStr.Replace("\"", "\\\"")}\"");
                                }
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: Failed to serialize metadata value for key '{entry.Key}': {ex.Message}");
                                sb.Append("\"SerializationError\"");
                            }
                        }
                        else
                        {
                            sb.Append("null");
                        }

                        first = false;
                    }

                    sb.Append("}");
                    return sb.ToString();
                }

                // Helper function to serialize float arrays
                byte[] SerializeFloatArray(float[] array)
                {
                    if (array == null) return new byte[0];
                    byte[] bytes = new byte[array.Length * sizeof(float)];
                    System.Buffer.BlockCopy(array, 0, bytes, 0, bytes.Length);
                    return bytes;
                }

                // Helper function to deserialize float arrays
                float[] DeserializeFloatArray(byte[] bytes)
                {
                    if (bytes == null || bytes.Length == 0) return new float[0];
                    float[] array = new float[bytes.Length / sizeof(float)];
                    System.Buffer.BlockCopy(bytes, 0, array, 0, bytes.Length);
                    return array;
                }

                // Helper function to process an array with K-means clustering - SEPARATED INPUT DATA SECTION
                void ProcessArrayWithKMeans(double[] dataArray, string arrayName, ConcurrentDictionary<string, object> resultsStore)
                {
                    // === CLUSTER INPUT DATA SECTION ===
                    // This data will be dynamically generated in future iterations
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] === CLUSTER INPUT DATA PROCESSING: {arrayName} ===");

                    if (dataArray == null || dataArray.Length < 3 || dataArray.All(d => d == dataArray[0]))
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Not enough distinct data points in {arrayName} for K-means clustering or data is constant. Skipping.");
                        resultsStore[$"{arrayName}_Category"] = "InsufficientData";
                        resultsStore[$"{arrayName}_NormalizedValue"] = 0.0;
                        resultsStore[$"{arrayName}_NormalizedX"] = 0.0;
                        resultsStore[$"{arrayName}_NormalizedY"] = 0.0;
                        resultsStore[$"{arrayName}_NormalizedZ"] = 0.0;
                        return;
                    }

                    try
                    {
                        // Convert 1D array to 2D array format required by Accord
                        double[][] points = new double[dataArray.Length][];
                        for (int i = 0; i < dataArray.Length; i++)
                        {
                            points[i] = new double[] { dataArray[i] };
                        }

                        // Initialize K-means algorithm with k=3 clusters
                        int k = Math.Min(3, points.Length);
                        if (k < 1) k = 1;

                        var kmeans = new Accord.MachineLearning.KMeans(k);
                        kmeans.Distance = new Accord.Math.Distances.SquareEuclidean();

                        try
                        {
                            var clusters = kmeans.Learn(points);
                            double[] centroids = clusters.Centroids.Select(c => c[0]).ToArray();

                            Array.Sort(centroids);
                            Array.Reverse(centroids);
                            int numCentroids = Math.Min(3, centroids.Length);

                            double[] paddedCentroids = new double[3];
                            for (int i = 0; i < numCentroids; i++) paddedCentroids[i] = centroids[i];

                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] K-means centroids for {arrayName}: [{string.Join(", ", centroids.Take(numCentroids).Select(c => c.ToString("F4")))}]");

                            double centralPoint = centroids.Take(numCentroids).DefaultIfEmpty(0).Average();
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Central point for {arrayName}: {centralPoint}");

                            double maxAbsCentroid = centroids.Take(numCentroids).Select(Math.Abs).DefaultIfEmpty(0).Max();
                            double normalizedValue = maxAbsCentroid > 1e-6 ? centralPoint / maxAbsCentroid : 0;

                            string category;
                            if (normalizedValue < -0.33) category = "Negative High";
                            else if (normalizedValue < 0) category = "Negative Low";
                            else if (normalizedValue < 0.33) category = "Positive Low";
                            else if (normalizedValue < 0.66) category = "Positive Medium";
                            else category = "Positive High";

                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Normalized value for {arrayName}: {normalizedValue:F4}, Category: {category}");

                            double x = paddedCentroids[0];
                            double y = paddedCentroids[1];
                            double z = paddedCentroids[2];

                            double maxAbsCoordinate = Math.Max(Math.Max(Math.Abs(x), Math.Abs(y)), Math.Abs(z));
                            if (maxAbsCoordinate > 1e-6)
                            {
                                x /= maxAbsCoordinate;
                                y /= maxAbsCoordinate;
                                z /= maxAbsCoordinate;
                            }

                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Normalized XYZ coordinates for {arrayName}: ({x:F4}, {y:F4}, {z:F4})");

                            resultsStore[$"{arrayName}_Category"] = category;
                            resultsStore[$"{arrayName}_NormalizedValue"] = normalizedValue;
                            resultsStore[$"{arrayName}_NormalizedX"] = x;
                            resultsStore[$"{arrayName}_NormalizedY"] = y;
                            resultsStore[$"{arrayName}_NormalizedZ"] = z;
                        }
                        catch (Exception learnEx)
                        {
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Error during K-means Learn for {arrayName}: {learnEx.Message}");
                            resultsStore[$"{arrayName}_Category"] = "ClusteringLearnError";
                            resultsStore[$"{arrayName}_NormalizedValue"] = 0.0;
                            resultsStore[$"{arrayName}_NormalizedX"] = 0.0;
                            resultsStore[$"{arrayName}_NormalizedY"] = 0.0;
                            resultsStore[$"{arrayName}_NormalizedZ"] = 0.0;
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Error processing K-means for {arrayName}: {ex.Message}");
                        resultsStore[$"{arrayName}_Category"] = "ProcessingError";
                        resultsStore[$"{arrayName}_NormalizedValue"] = 0.0;
                        resultsStore[$"{arrayName}_NormalizedX"] = 0.0;
                        resultsStore[$"{arrayName}_NormalizedY"] = 0.0;
                        resultsStore[$"{arrayName}_NormalizedZ"] = 0.0;
                    }
                }

                // Helper function to calculate trajectory stability
                double CalculateTrajectoryStability(List<double[]> trajectoryPoints)
                {
                    if (trajectoryPoints == null || trajectoryPoints.Count < 2)
                        return 0.5;

                    double averageAngleChange = 0;
                    int angleCount = 0;

                    for (int i = 1; i < trajectoryPoints.Count - 1; i++)
                    {
                        if (trajectoryPoints[i - 1] == null || trajectoryPoints[i - 1].Length < 3 ||
                           trajectoryPoints[i] == null || trajectoryPoints[i].Length < 3 ||
                           trajectoryPoints[i + 1] == null || trajectoryPoints[i + 1].Length < 3)
                        {
                            continue;
                        }

                        double[] prevVector = new double[3];
                        double[] nextVector = new double[3];

                        for (int j = 0; j < 3; j++)
                            prevVector[j] = trajectoryPoints[i][j] - trajectoryPoints[i - 1][j];

                        for (int j = 0; j < 3; j++)
                            nextVector[j] = trajectoryPoints[i + 1][j] - trajectoryPoints[i][j];

                        double dotProduct = 0;
                        double prevMagSq = 0;
                        double nextMagSq = 0;

                        for (int j = 0; j < 3; j++)
                        {
                            dotProduct += prevVector[j] * nextVector[j];
                            prevMagSq += prevVector[j] * prevVector[j];
                            nextMagSq += nextVector[j] * nextVector[j];
                        }

                        double prevMag = Math.Sqrt(prevMagSq);
                        double nextMag = Math.Sqrt(nextMagSq);

                        if (prevMag > 1e-9 && nextMag > 1e-9)
                        {
                            double cosAngle = dotProduct / (prevMag * nextMag);
                            cosAngle = Math.Max(-1.0, Math.Min(1.0, cosAngle));
                            double angle = Math.Acos(cosAngle);
                            averageAngleChange += angle;
                            angleCount++;
                        }
                    }

                    if (angleCount > 0)
                        averageAngleChange /= angleCount;
                    else
                        return 0.5;

                    double stabilityScore = 1.0 - (averageAngleChange / Math.PI);
                    return stabilityScore;
                }

                // Helper function to calculate plane intersection point
                double[] CalculatePlaneIntersection(List<double[]> trajectoryPoints, int planeAxis, double tolerance)
                {
                    if (trajectoryPoints == null || trajectoryPoints.Count < 2)
                        return null;

                    for (int i = 1; i < trajectoryPoints.Count; i++)
                    {
                        if (trajectoryPoints[i - 1] == null || trajectoryPoints[i - 1].Length < Math.Max(3, planeAxis + 1) ||
                           trajectoryPoints[i] == null || trajectoryPoints[i].Length < Math.Max(3, planeAxis + 1))
                        {
                            continue;
                        }

                        double v1 = trajectoryPoints[i - 1][planeAxis];
                        double v2 = trajectoryPoints[i][planeAxis];

                        bool crossedZero = (v1 >= -tolerance && v2 <= tolerance) || (v1 <= tolerance && v2 >= -tolerance);

                        if (crossedZero && !((v1 >= -tolerance && v1 <= tolerance) && (v2 >= -tolerance && v2 <= tolerance)))
                        {
                            if (Math.Abs(v1) < tolerance) return (double[])trajectoryPoints[i - 1].Clone();
                            if (Math.Abs(v2) < tolerance) return (double[])trajectoryPoints[i].Clone();

                            double t = Math.Abs(v1) / (Math.Abs(v1) + Math.Abs(v2));

                            double[] intersection = new double[3];
                            for (int j = 0; j < 3; j++)
                            {
                                if (j < trajectoryPoints[i - 1].Length && j < trajectoryPoints[i].Length)
                                {
                                    intersection[j] = trajectoryPoints[i - 1][j] * (1 - t) + trajectoryPoints[i][j] * t;
                                }
                                else
                                {
                                    intersection[j] = 0;
                                }
                            }

                            if (planeAxis >= 0 && planeAxis < intersection.Length)
                            {
                                intersection[planeAxis] = 0.0;
                            }

                            return intersection;
                        }
                    }

                    return null;
                }

                // Helper functions to count negative points using tolerance
                int CountNegativePoints(List<double[]> points, int axis, double tolerance)
                {
                    if (points == null) return 0;
                    int count = 0;
                    foreach (var point in points)
                    {
                        if (point != null && point.Length > axis && point[axis] < -tolerance)
                            count++;
                    }
                    return count;
                }

                int CountNegativeBothPoints(List<double[]> points, double tolerance)
                {
                    if (points == null) return 0;
                    int count = 0;
                    foreach (var point in points)
                    {
                        if (point != null && point.Length >= 2 && point[0] < -tolerance && point[1] < -tolerance)
                            count++;
                    }
                    return count;
                }

                double[] FindPositiveCoordinate(List<double[]> points, double tolerance)
                {
                    if (points == null || points.Count == 0) return new double[] { 0, 0, 0 };

                    foreach (var point in points)
                    {
                        if (point != null && point.Length >= 2 && point[0] > tolerance && point[1] > tolerance)
                            return (double[])point.Clone();
                    }

                    int firstIndex = points.Count > 0 ? 0 : -1;
                    if (firstIndex != -1 && points[firstIndex] != null && points[firstIndex].Length >= 3)
                        return (double[])points[firstIndex].Clone();
                    return new double[] { 0, 0, 0 };
                }

                double[] FindNegativeCoordinate(List<double[]> points, double tolerance)
                {
                    if (points == null || points.Count == 0) return new double[] { 0, 0, 0 };

                    foreach (var point in points)
                    {
                        if (point != null && point.Length >= 2 && point[0] < -tolerance && point[1] < -tolerance)
                            return (double[])point.Clone();
                    }

                    int lastIndex = points.Count > 0 ? points.Count - 1 : 0;
                    if (lastIndex >= 0 && points[lastIndex] != null && points[lastIndex].Length >= 3)
                        return (double[])points[lastIndex].Clone();
                    return new double[] { 0, 0, 0 };
                }

                double CalculateVelocity(double[] trajectory, double magnitude)
                {
                    return magnitude;
                }

                //==========================================================================
                // Workflow Coordination
                //==========================================================================
                string ExecuteProductionWorkflow(CoreMlOutcomeRecord record, int custId, Tensorflow.Session session_param_unused)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Starting multi-stage workflow for customer {custId}.");

                    // Step 1: Begin Data Acquisition and Initial Feature Analysis
                    string analysisResult = Stage1_DataAcquisitionAndAnalysis(record, custId);
                    unitResultsStore["DataAcquisitionResult"] = analysisResult;

                    // Step 2: Begin Feature Tensor Generation and Trajectory Mapping
                    string tensorMappingResult = Stage2_FeatureTensorAndMapping(analysisResult, custId);
                    unitResultsStore["FeatureTensorMappingResult"] = tensorMappingResult;

                    // Step 3: Begin Processed Feature Definition Creation
                    string processedFeatureResult = Stage3_ProcessedFeatureDefinition(tensorMappingResult, custId);
                    unitResultsStore["ProcessedFeatureResult"] = processedFeatureResult;

                    // Step 4: Begin Feature Quality Assessment
                    string qualityAssessmentResult = Stage4_FeatureQualityAssessment(processedFeatureResult, custId);
                    unitResultsStore["QualityAssessmentResult"] = qualityAssessmentResult;

                    // Step 5: Begin Combined Feature Evaluation
                    float combinedEvaluationScore = Stage5_CombinedFeatureEvaluation(qualityAssessmentResult, custId);
                    unitResultsStore["CombinedEvaluationScore"] = combinedEvaluationScore;

                    // Step 6: Begin Fractal Optimization Analysis
                    string optimizationAnalysisResult = Stage6_FractalOptimizationAnalysis(qualityAssessmentResult, combinedEvaluationScore, custId);
                    unitResultsStore["OptimizationAnalysisResult"] = optimizationAnalysisResult;

                    // Step 7: Begin Tensor Network Training with Curvature Embedding (Includes Actual TF.NET)
                    // PROLIFERATION OCCURS HERE: During training iterations, the "1+P" expression proliferates
                    string trainingOutcomeResult = Stage7_TensorNetworkTraining(optimizationAnalysisResult, custId, unitResultsStore);
                    unitResultsStore["TensorNetworkTrainingOutcome"] = trainingOutcomeResult;

                    // Step 8: Begin Future Performance Projection
                    string performanceProjectionResult = Stage8_FutureProjection(trainingOutcomeResult, combinedEvaluationScore, custId);
                    unitResultsStore["PerformanceProjectionResult"] = performanceProjectionResult;

                    float finalScore = unitResultsStore.TryGetValue("ProjectedPerformanceScore", out var projectedScoreVal)
                        ? Convert.ToSingle(projectedScoreVal) : combinedEvaluationScore;

                    unitResultsStore["ModelAProcessingOutcome"] = finalScore;
                    unitResultsStore["A_FinalScore"] = finalScore;

                    if (!unitResultsStore.ContainsKey("ModelACombinedParameters") && RuntimeProcessingContext.RetrieveContextValue("model_a_params_combined") is byte[] params_combined)
                    {
                        unitResultsStore["ModelACombinedParameters"] = params_combined;
                    }

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Workflow completed for customer {custId} with final score {finalScore:F4}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: ModelAProcessingOutcome stored: {unitResultsStore.ContainsKey("ModelAProcessingOutcome")}");

                    return $"Workflow_Complete_Cust_{custId}_FinalScore_{finalScore:F4}";
                }

                //==========================================================================
                // Step 1: Data Acquisition & Analysis
                //==========================================================================
                string Stage1_DataAcquisitionAndAnalysis(CoreMlOutcomeRecord record, int custId)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 1 - Acquiring data and analyzing initial features for customer {custId}.");

                    var productInventory = RuntimeProcessingContext.RetrieveContextValue("All_Simulated_Product_Inventory") as List<dynamic>;
                    var serviceOfferings = RuntimeProcessingContext.RetrieveContextValue("All_Simulated_Service_Offerings") as List<dynamic>;

                    if (productInventory != null)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Step 1 - Processing Product Data ({productInventory.Count} items).");
                        var quantityAvailable = new List<int>();
                        var productMonetaryValue = new List<double>();
                        var productCostContribution = new List<double>();

                        foreach (var product in productInventory)
                        {
                            try
                            {
                                quantityAvailable.Add(Convert.ToInt32(product.QuantityAvailable));
                                productMonetaryValue.Add(Convert.ToDouble(product.MonetaryValue));
                                productCostContribution.Add(Convert.ToDouble(product.CostContributionValue));
                            }
                            catch (Microsoft.CSharp.RuntimeBinder.RuntimeBinderException rbEx)
                            {
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] RuntimeBinder Error accessing product properties: {rbEx.Message}");
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unexpected Error accessing product properties: {ex.Message}");
                            }
                        }

                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product QuantityAvailable: [{string.Join(", ", quantityAvailable)}]");
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product MonetaryValue: [{string.Join(", ", productMonetaryValue)}]");
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product CostContributionValue: [{string.Join(", ", productCostContribution)}]");

                        ProcessArrayWithKMeans(quantityAvailable.Select(x => (double)x).ToArray(), "Product QuantityAvailable", unitResultsStore);
                        ProcessArrayWithKMeans(productMonetaryValue.ToArray(), "Product MonetaryValue", unitResultsStore);
                        ProcessArrayWithKMeans(productCostContribution.ToArray(), "Product CostContributionValue", unitResultsStore);
                    }
                    else
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product inventory data not found in RuntimeProcessingContext. Skipping product analysis.");
                    }

                    if (serviceOfferings != null)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Step 1 - Processing Service Data ({serviceOfferings.Count} items).");
                        var fulfillmentQuantity = new List<int>();
                        var serviceMonetaryValue = new List<double>();
                        var serviceCostContribution = new List<double>();

                        foreach (var service in serviceOfferings)
                        {
                            try
                            {
                                fulfillmentQuantity.Add(Convert.ToInt32(service.FulfillmentQuantity));
                                serviceMonetaryValue.Add(Convert.ToDouble(service.MonetaryValue));
                                serviceCostContribution.Add(Convert.ToDouble(service.CostContributionValue));
                            }
                            catch (Microsoft.CSharp.RuntimeBinder.RuntimeBinderException rbEx)
                            {
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] RuntimeBinder Error accessing service properties: {rbEx.Message}");
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unexpected Error accessing service properties: {ex.Message}");
                            }
                        }

                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service FulfillmentQuantity: [{string.Join(", ", fulfillmentQuantity)}]");
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service MonetaryValue: [{string.Join(", ", serviceMonetaryValue)}]");
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service CostContributionValue: [{string.Join(", ", serviceCostContribution)}]");

                        ProcessArrayWithKMeans(fulfillmentQuantity.Select(x => (double)x).ToArray(), "Service FulfillmentQuantity", unitResultsStore);
                        ProcessArrayWithKMeans(serviceMonetaryValue.ToArray(), "Service MonetaryValue", unitResultsStore);
                        ProcessArrayWithKMeans(serviceCostContribution.ToArray(), "Service CostContributionValue", unitResultsStore);
                    }
                    else
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service offerings data not found in RuntimeProcessingContext. Skipping service analysis.");
                    }

                    string result = $"InitialAnalysis_Cust_{custId}_Record_{record.RecordIdentifier}";
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 1 - Data acquisition and initial analysis completed: {result}");
                    return result;
                }

                //==========================================================================
                // Step 2: Feature Tensor Generation & Mapping
                //==========================================================================
                string Stage2_FeatureTensorAndMapping(string analysisResult, int custId)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 2 - Generating feature tensors and mapping trajectories for customer {custId}.");

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Step 2 - Retrieving coordinates from Step 1 analysis.");

                    //------------------------------------------
                    // PRODUCT COORDINATES (Derived from Step 1 K-means analysis)
                    //------------------------------------------
                    string prodQtyCategory = unitResultsStore.TryGetValue("Product QuantityAvailable_Category", out var pqc) ? pqc.ToString() : "Unknown";
                    double prodQtyX = unitResultsStore.TryGetValue("Product QuantityAvailable_NormalizedX", out var pqx) ? Convert.ToDouble(pqx) : 0;
                    double prodQtyY = unitResultsStore.TryGetValue("Product QuantityAvailable_NormalizedY", out var pqy) ? Convert.ToDouble(pqy) : 0;
                    double prodQtyZ = unitResultsStore.TryGetValue("Product QuantityAvailable_NormalizedZ", out var pqz) ? Convert.ToDouble(pqz) : 0;

                    string prodMonCategory = unitResultsStore.TryGetValue("Product MonetaryValue_Category", out var pmc) ? pmc.ToString() : "Unknown";
                    double prodMonX = unitResultsStore.TryGetValue("Product MonetaryValue_NormalizedX", out var pmx) ? Convert.ToDouble(pmx) : 0;
                    double prodMonY = unitResultsStore.TryGetValue("Product MonetaryValue_NormalizedY", out var pmy) ? Convert.ToDouble(pmy) : 0;
                    double prodMonZ = unitResultsStore.TryGetValue("Product MonetaryValue_NormalizedZ", out var pmz) ? Convert.ToDouble(pmz) : 0;

                    string prodCostCategory = unitResultsStore.TryGetValue("Product CostContributionValue_Category", out var pcc) ? pcc.ToString() : "Unknown";
                    double prodCostX = unitResultsStore.TryGetValue("Product CostContributionValue_NormalizedX", out var pcx) ? Convert.ToDouble(pcx) : 0;
                    double prodCostY = unitResultsStore.TryGetValue("Product CostContributionValue_NormalizedY", out var pcy) ? Convert.ToDouble(pcy) : 0;
                    double prodCostZ = unitResultsStore.TryGetValue("Product CostContributionValue_NormalizedZ", out var pcz) ? Convert.ToDouble(pcz) : 0;

                    //------------------------------------------
                    // SERVICE COORDINATES (Derived from Step 1 K-means analysis)
                    //------------------------------------------
                    string servFulfillCategory = unitResultsStore.TryGetValue("Service FulfillmentQuantity_Category", out var sfc) ? sfc.ToString() : "Unknown";
                    double servFulfillX = unitResultsStore.TryGetValue("Service FulfillmentQuantity_NormalizedX", out var sfx) ? Convert.ToDouble(sfx) : 0;
                    double servFulfillY = unitResultsStore.TryGetValue("Service FulfillmentQuantity_NormalizedY", out var sfy) ? Convert.ToDouble(sfy) : 0;
                    double servFulfillZ = unitResultsStore.TryGetValue("Service FulfillmentQuantity_NormalizedZ", out var sfz) ? Convert.ToDouble(sfz) : 0;

                    string servMonCategory = unitResultsStore.TryGetValue("Service MonetaryValue_Category", out var smc) ? smc.ToString() : "Unknown";
                    double servMonX = unitResultsStore.TryGetValue("Service MonetaryValue_NormalizedX", out var smx) ? Convert.ToDouble(smx) : 0;
                    double servMonY = unitResultsStore.TryGetValue("Service MonetaryValue_NormalizedY", out var smy) ? Convert.ToDouble(smy) : 0;
                    double servMonZ = unitResultsStore.TryGetValue("Service MonetaryValue_NormalizedZ", out var smz) ? Convert.ToDouble(smz) : 0;

                    string servCostCategory = unitResultsStore.TryGetValue("Service CostContributionValue_Category", out var scc) ? scc.ToString() : "Unknown";
                    double servCostX = unitResultsStore.TryGetValue("Service CostContributionValue_NormalizedX", out var scx) ? Convert.ToDouble(scx) : 0;
                    double servCostY = unitResultsStore.TryGetValue("Service CostContributionValue_NormalizedY", out var scy) ? Convert.ToDouble(scy) : 0;
                    double servCostZ = unitResultsStore.TryGetValue("Service CostContributionValue_NormalizedZ", out var scz) ? Convert.ToDouble(scz) : 0;

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Step 2 - Calculating tensors, magnitudes, and trajectories.");

                    //------------------------------------------
                    // PRODUCT TENSOR CALCULATIONS
                    //------------------------------------------
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] ----- PRODUCT TENSOR AND MAGNITUDE CALCULATIONS -----");

                    double prodOverallTensorX = (prodQtyX + prodMonX + prodCostX) / 3.0;
                    double prodOverallTensorY = (prodQtyY + prodMonY + prodCostY) / 3.0;
                    double prodOverallTensorZ = (prodQtyZ + prodMonZ + prodCostZ) / 3.0;
                    double prodOverallMagnitude = Math.Sqrt(prodOverallTensorX * prodOverallTensorX + prodOverallTensorY * prodOverallTensorY + prodOverallTensorZ * prodOverallTensorZ);

                    double[] prodTrajectory = new double[3] { 0, 0, 0 };
                    if (prodOverallMagnitude > 1e-9)
                    {
                        prodTrajectory[0] = prodOverallTensorX / prodOverallMagnitude;
                        prodTrajectory[1] = prodOverallTensorY / prodOverallMagnitude;
                        prodTrajectory[2] = prodOverallTensorZ / prodOverallMagnitude;
                    }

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product Overall Tensor: ({prodOverallTensorX:F4}, {prodOverallTensorY:F4}, {prodOverallTensorZ:F4})");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product Overall Magnitude: {prodOverallMagnitude:F4}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product Trajectory: ({prodTrajectory[0]:F4}, {prodTrajectory[1]:F4}, {prodTrajectory[2]:F4})");

                    //------------------------------------------
                    // SERVICE TENSOR CALCULATIONS
                    //------------------------------------------
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] ----- SERVICE TENSOR AND MAGNITUDE CALCULATIONS -----");

                    double servOverallTensorX = (servFulfillX + servMonX + servCostX) / 3.0;
                    double servOverallTensorY = (servFulfillY + servMonY + servCostY) / 3.0;
                    double servOverallTensorZ = (servFulfillZ + servMonZ + servCostZ) / 3.0;
                    double servOverallMagnitude = Math.Sqrt(servOverallTensorX * servOverallTensorX + servOverallTensorY * servOverallTensorY + servOverallTensorZ * servOverallTensorZ);

                    double[] servTrajectory = new double[3] { 0, 0, 0 };
                    if (servOverallMagnitude > 1e-9)
                    {
                        servTrajectory[0] = servOverallTensorX / servOverallMagnitude;
                        servTrajectory[1] = servOverallTensorY / servOverallMagnitude;
                        servTrajectory[2] = servOverallTensorZ / servOverallMagnitude;
                    }

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service Overall Tensor: ({servOverallTensorX:F4}, {servOverallTensorY:F4}, {servOverallTensorZ:F4})");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service Overall Magnitude: {servOverallMagnitude:F4}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service Trajectory: ({servTrajectory[0]:F4}, {servTrajectory[1]:F4}, {servTrajectory[2]:F4})");

                    //==========================================================================
                    // Trajectory Plot Generation & Analysis
                    //==========================================================================
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] ----- TRAJECTORY PLOT GENERATION & ANALYSIS -----");

                    const int MAX_RECURSION_DEPTH = 35;
                    const double CONTINUE_PAST_PLANE = -2.0;

                    List<double[]> productTrajectoryPoints = new List<double[]>();
                    List<double> productPointIntensities = new List<double>();
                    List<double[]> serviceTrajectoryPoints = new List<double[]>();
                    List<double> servicePointIntensities = new List<double>();

                    double[] productCurrentPosition = new double[] { prodOverallTensorX, prodOverallTensorY, prodOverallTensorZ };
                    double[] serviceCurrentPosition = new double[] { servOverallTensorX, servOverallTensorY, servOverallTensorZ };

                    double recursionFactor = 0.95;

                    double[] InvertTrajectoryIfNeeded(double[] trajectory)
                    {
                        bool movesTowardNegativeX = trajectory != null && trajectory.Length > 0 && trajectory[0] < -1e-6;
                        bool movesTowardNegativeY = trajectory != null && trajectory.Length > 1 && trajectory[1] < -1e-6;

                        if (!movesTowardNegativeX || !movesTowardNegativeY)
                        {
                            if (trajectory == null || trajectory.Length < 3) return new double[] { 0, 0, 0 };

                            double[] invertedTrajectory = new double[3];
                            invertedTrajectory[0] = movesTowardNegativeX ? trajectory[0] : -Math.Abs(trajectory[0]);
                            invertedTrajectory[1] = movesTowardNegativeY ? trajectory[1] : -Math.Abs(trajectory[1]);
                            invertedTrajectory[2] = trajectory.Length > 2 ? trajectory[2] : 0;

                            double magnitude = Math.Sqrt(
                                invertedTrajectory[0] * invertedTrajectory[0] +
                                invertedTrajectory[1] * invertedTrajectory[1] +
                                invertedTrajectory[2] * invertedTrajectory[2]
                            );

                            if (magnitude > 1e-9)
                            {
                                invertedTrajectory[0] /= magnitude;
                                invertedTrajectory[1] /= magnitude;
                                invertedTrajectory[2] /= magnitude;
                            }

                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Inverted trajectory from ({trajectory[0]:F4}, {trajectory[1]:F4}, {trajectory[2]:F4}) to ({invertedTrajectory[0]:F4}, {invertedTrajectory[1]:F4}, {invertedTrajectory[2]:F4})");
                            return invertedTrajectory;
                        }
                        if (trajectory != null && trajectory.Length >= 3) return (double[])trajectory.Clone();
                        return new double[] { 0, 0, 0 };
                    }

                    double[] productTrajectoryAdjusted = InvertTrajectoryIfNeeded(prodTrajectory);
                    double[] serviceTrajectoryAdjusted = InvertTrajectoryIfNeeded(servTrajectory);

                    void RecursivePlotTrajectory(double[] currentPosition, double[] trajectory, double magnitude,
                                                List<double[]> points, List<double> intensities, int depth,
                                                string trajectoryName)
                    {
                        if (currentPosition == null || trajectory == null || currentPosition.Length < 3 || trajectory.Length < 3) return;

                        if (depth >= MAX_RECURSION_DEPTH)
                        {
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] {trajectoryName} recursion stopped at max depth {depth}");
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] {trajectoryName} final position: ({currentPosition[0]:F6}, {currentPosition[1]:F6}, {currentPosition[2]:F4})");
                            points.Add((double[])currentPosition.Clone());
                            double finalPointIntensity = magnitude * Math.Pow(recursionFactor, depth);
                            intensities.Add(finalPointIntensity);
                            return;
                        }

                        if (currentPosition[0] < CONTINUE_PAST_PLANE && currentPosition[1] < CONTINUE_PAST_PLANE)
                        {
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] {trajectoryName} recursion stopped - Reached target negative threshold at depth {depth}");
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] {trajectoryName} final position: ({currentPosition[0]:F6}, {currentPosition[1]:F6}, {currentPosition[2]:F4})");
                            points.Add((double[])currentPosition.Clone());
                            double finalPointIntensity = magnitude * Math.Pow(recursionFactor, depth);
                            intensities.Add(finalPointIntensity);
                            return;
                        }

                        points.Add((double[])currentPosition.Clone());

                        double currentPointIntensity = magnitude * Math.Pow(recursionFactor, depth);
                        intensities.Add(currentPointIntensity);

                        bool beyondXPlane = currentPosition[0] < -1e-6;
                        bool beyondYPlane = currentPosition[1] < -1e-6;
                        bool beyondBothPlanes = beyondXPlane && beyondYPlane;

                        if (depth % 4 == 0 || beyondBothPlanes ||
                            (depth > 0 && points.Count > 1 && (
                                (points[points.Count - 2][0] >= -1e-6 && beyondXPlane) ||
                                (points[points.Count - 2][1] >= -1e-6 && beyondYPlane)
                            )))
                        {
                            string positionInfo = "";
                            if (beyondXPlane) positionInfo += " BEYOND-X-PLANE";
                            if (beyondYPlane) positionInfo += " BEYOND-Y-PLANE";

                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] {trajectoryName} point {depth}: " +
                                             $"Position=({currentPosition[0]:F6}, {currentPosition[1]:F6}, {currentPosition[2]:F4}), " +
                                             $"Intensity={currentPointIntensity:F4}{positionInfo}");
                        }

                        double stepMultiplier = 1.0;

                        if (depth < 10)
                        {
                            stepMultiplier = 2.0;
                        }
                        else if (!beyondBothPlanes && depth < MAX_RECURSION_DEPTH - 5)
                        {
                            stepMultiplier = 1.5;
                        }
                        else
                        {
                            stepMultiplier = 1.0;
                        }

                        double stepSize = magnitude * Math.Pow(recursionFactor, depth) * 0.4 * stepMultiplier;

                        double[] nextPosition = new double[3];
                        for (int i = 0; i < 3; i++)
                        {
                            nextPosition[i] = currentPosition[i] + trajectory[i] * stepSize;
                        }

                        RecursivePlotTrajectory(nextPosition, trajectory, magnitude, points, intensities, depth + 1, trajectoryName);
                    }

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Generating Product trajectory recursive plot");
                    RecursivePlotTrajectory(productCurrentPosition, productTrajectoryAdjusted, prodOverallMagnitude,
                                           productTrajectoryPoints, productPointIntensities, 0, "PRODUCT");

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Generating Service trajectory recursive plot");
                    RecursivePlotTrajectory(serviceCurrentPosition, serviceTrajectoryAdjusted, servOverallMagnitude,
                                           serviceTrajectoryPoints, servicePointIntensities, 0, "SERVICE");

                    //==========================================================================
                    // Trajectory Intersection Analysis
                    //==========================================================================

                    double[] productXPlaneIntersection = CalculatePlaneIntersection(productTrajectoryPoints, 0, 1e-6);
                    double[] productYPlaneIntersection = CalculatePlaneIntersection(productTrajectoryPoints, 1, 1e-6);
                    double[] serviceXPlaneIntersection = CalculatePlaneIntersection(serviceTrajectoryPoints, 0, 1e-6);
                    double[] serviceYPlaneIntersection = CalculatePlaneIntersection(serviceTrajectoryPoints, 1, 1e-6);

                    //==========================================================================
                    // Feature Coordinate Extraction
                    //==========================================================================

                    //------------------------------------------
                    // PRODUCT TRAJECTORY VARIABLES
                    //------------------------------------------
                    double[] productVector = (productTrajectoryAdjusted != null && productTrajectoryAdjusted.Length >= 3) ? (double[])productTrajectoryAdjusted.Clone() : new double[] { 0, 0, 0 };
                    double productVelocity = CalculateVelocity(productVector, prodOverallMagnitude);
                    double[] productPositiveCoordinate = FindPositiveCoordinate(productTrajectoryPoints, 1e-6);
                    double[] productNegativeCoordinate = FindNegativeCoordinate(productTrajectoryPoints, 1e-6);

                    //------------------------------------------
                    // SERVICE TRAJECTORY VARIABLES
                    //------------------------------------------
                    double[] serviceVector = (serviceTrajectoryAdjusted != null && serviceTrajectoryAdjusted.Length >= 3) ? (double[])serviceTrajectoryAdjusted.Clone() : new double[] { 0, 0, 0 };
                    double serviceVelocity = CalculateVelocity(serviceVector, servOverallMagnitude);
                    double[] servicePositiveCoordinate = FindPositiveCoordinate(serviceTrajectoryPoints, 1e-6);
                    double[] serviceNegativeCoordinate = FindNegativeCoordinate(serviceTrajectoryPoints, 1e-6);

                    //==========================================================================
                    // Trajectory Analysis Logging
                    //==========================================================================

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] ----- PLANE INTERSECTION ANALYSIS -----");

                    if (productXPlaneIntersection != null)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product X-Plane Intersection: " +
                                         $"(0.000000, {productXPlaneIntersection[1]:F6}, {productXPlaneIntersection[2]:F6})");
                    }
                    else
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product trajectory does not intersect X-Plane");
                    }

                    if (productYPlaneIntersection != null)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product Y-Plane Intersection: " +
                                         $"({productYPlaneIntersection[0]:F6}, 0.000000, {productYPlaneIntersection[2]:F6})");
                    }
                    else
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product trajectory does not intersect Y-Plane");
                    }

                    if (serviceXPlaneIntersection != null)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service X-Plane Intersection: " +
                                         $"(0.000000, {serviceXPlaneIntersection[1]:F6}, {serviceXPlaneIntersection[2]:F6})");
                    }
                    else
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service trajectory does not intersect X-Plane");
                    }

                    if (serviceYPlaneIntersection != null)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service Y-Plane Intersection: " +
                                         $"({serviceYPlaneIntersection[0]:F6}, 0.000000, {serviceYPlaneIntersection[2]:F6})");
                    }
                    else
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service trajectory does not intersect Y-Plane");
                    }

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] ----- KEY TRAJECTORY DATA -----");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product Vector: ({productVector[0]:F6}, {productVector[1]:F6}, {productVector[2]:F6})");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product Velocity: {productVelocity:F6}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product Positive Coordinate: ({productPositiveCoordinate[0]:F6}, {productPositiveCoordinate[1]:F6}, {productPositiveCoordinate[2]:F6})");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product Negative Coordinate: ({productNegativeCoordinate[0]:F6}, {productNegativeCoordinate[1]:F6}, {productNegativeCoordinate[2]:F6})");

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service Vector: ({serviceVector[0]:F6}, {serviceVector[1]:F6}, {serviceVector[2]:F6})");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service Velocity: {serviceVelocity:F6}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service Positive Coordinate: ({servicePositiveCoordinate[0]:F6}, {servicePositiveCoordinate[1]:F6}, {servicePositiveCoordinate[2]:F6})");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service Negative Coordinate: ({serviceNegativeCoordinate[0]:F6}, {serviceNegativeCoordinate[1]:F6}, {serviceNegativeCoordinate[2]:F6})");

                    //==========================================================================
                    // Trajectory Statistics
                    //==========================================================================

                    int productNegativeXCount = CountNegativePoints(productTrajectoryPoints, 0, 1e-6);
                    int productNegativeYCount = CountNegativePoints(productTrajectoryPoints, 1, 1e-6);
                    int productNegativeBothCount = CountNegativeBothPoints(productTrajectoryPoints, 1e-6);

                    int serviceNegativeXCount = CountNegativePoints(serviceTrajectoryPoints, 0, 1e-6);
                    int serviceNegativeYCount = CountNegativePoints(serviceTrajectoryPoints, 1, 1e-6);
                    int serviceNegativeBothCount = CountNegativeBothPoints(serviceTrajectoryPoints, 1e-6);

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product negative X count: {productNegativeXCount}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product negative Y count: {productNegativeYCount}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product negative both count: {productNegativeBothCount}");

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service negative X count: {serviceNegativeXCount}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service negative Y count: {serviceNegativeYCount}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service negative both count: {serviceNegativeBothCount}");

                    //==========================================================================
                    // Store Analysis Results
                    //==========================================================================

                    unitResultsStore["Product_TrajectoryPoints"] = productTrajectoryPoints;
                    unitResultsStore["Product_PointIntensities"] = productPointIntensities;
                    unitResultsStore["Service_TrajectoryPoints"] = serviceTrajectoryPoints;
                    unitResultsStore["Service_PointIntensities"] = servicePointIntensities;

                    unitResultsStore["Product_XPlaneIntersection"] = productXPlaneIntersection;
                    unitResultsStore["Product_YPlaneIntersection"] = productYPlaneIntersection;
                    unitResultsStore["Service_XPlaneIntersection"] = serviceXPlaneIntersection;
                    unitResultsStore["Service_YPlaneIntersection"] = serviceYPlaneIntersection;

                    unitResultsStore["Product_Vector"] = productVector;
                    unitResultsStore["Product_Velocity"] = productVelocity;
                    unitResultsStore["Product_PositiveCoordinate"] = productPositiveCoordinate;
                    unitResultsStore["Product_NegativeCoordinate"] = productNegativeCoordinate;

                    unitResultsStore["Service_Vector"] = serviceVector;
                    unitResultsStore["Service_Velocity"] = serviceVelocity;
                    unitResultsStore["Service_PositiveCoordinate"] = servicePositiveCoordinate;
                    unitResultsStore["Service_NegativeCoordinate"] = serviceNegativeCoordinate;

                    unitResultsStore["Product_NegativeXCount"] = productNegativeXCount;
                    unitResultsStore["Product_NegativeYCount"] = productNegativeYCount;
                    unitResultsStore["Product_NegativeBothCount"] = productNegativeBothCount;
                    unitResultsStore["Service_NegativeXCount"] = serviceNegativeXCount;
                    unitResultsStore["Service_NegativeYCount"] = serviceNegativeYCount;
                    unitResultsStore["Service_NegativeBothCount"] = serviceNegativeBothCount;

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product trajectory plot: {productTrajectoryPoints?.Count ?? 0} points, {productNegativeBothCount} in negative X-Y quadrant");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service trajectory plot: {serviceTrajectoryPoints?.Count ?? 0} points, {serviceNegativeBothCount} in negative X-Y quadrant");

                    string result = $"FeatureTensorsAndMapping_Cust_{custId}_BasedOn_{analysisResult.Replace("InitialAnalysis_", "")}";

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 2 - Feature tensor generation and mapping completed: {result}");
                    return result;
                }

                //==========================================================================
                // Step 3: Processed Feature Definition Creation
                //==========================================================================
                string Stage3_ProcessedFeatureDefinition(string tensorMappingResult, int custId)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 3 - Creating processed feature definition for customer {custId}.");

                    double productVelocity = unitResultsStore.TryGetValue("Product_Velocity", out var pv)
                        ? Convert.ToDouble(pv) : 0.5;
                    double serviceVelocity = unitResultsStore.TryGetValue("Service_Velocity", out var sv)
                        ? Convert.ToDouble(sv) : 0.5;

                    var productTrajectoryPoints = unitResultsStore.TryGetValue("Product_TrajectoryPoints", out var ptp)
                        ? ptp as List<double[]> : new List<double[]>();
                    var serviceTrajectoryPoints = unitResultsStore.TryGetValue("Service_TrajectoryPoints", out var stp)
                        ? stp as List<double[]> : new List<double[]>();

                    double[] productXPlaneIntersection = unitResultsStore.TryGetValue("Product_XPlaneIntersection", out var pxi)
                        ? pxi as double[] : null;
                    double[] productYPlaneIntersection = unitResultsStore.TryGetValue("Product_YPlaneIntersection", out var pyi)
                        ? pyi as double[] : null;

                    double velocityComponent = (productVelocity + serviceVelocity) / 2.0;
                    double trajectoryStability = 0.5;
                    double intersectionQuality = 0.5;

                    if (productTrajectoryPoints != null && productTrajectoryPoints.Count > 1)
                    {
                        trajectoryStability = CalculateTrajectoryStability(productTrajectoryPoints);
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] QA product trajectory stability: {trajectoryStability:F4}");
                    }

                    if (productXPlaneIntersection != null && productYPlaneIntersection != null)
                    {
                        double zDifference = Math.Abs(productXPlaneIntersection[2] - productYPlaneIntersection[2]);
                        intersectionQuality = 1.0 - Math.Min(1.0, zDifference);
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] QA intersection quality: {intersectionQuality:F4}");
                    }

                    double qaScore = velocityComponent * 0.4 + trajectoryStability * 0.3 + intersectionQuality * 0.3;
                    qaScore = Math.Min(qaScore, 1.0);

                    int qaLevel = (int)(qaScore * 3) + 1;
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] QA final score: {qaScore:F4}, level: {qaLevel}");

                    string result = $"QualityAssessment_Passed_Level_{qaLevel}_V{velocityComponent:F2}_S{trajectoryStability:F2}_I{intersectionQuality:F2}";

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 3 - Processed feature definition created: {result}");
                    return result;
                }

                //==========================================================================
                // Step 4: Feature Quality Assessment
                //==========================================================================
                string Stage4_FeatureQualityAssessment(string processedFeatureResult, int custId)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 4 - Assessing feature quality for customer {custId}.");

                    double productVelocity = unitResultsStore.TryGetValue("Product_Velocity", out var pv)
                        ? Convert.ToDouble(pv) : 0.5;
                    double serviceVelocity = unitResultsStore.TryGetValue("Service_Velocity", out var sv)
                        ? Convert.ToDouble(sv) : 0.5;

                    var productTrajectoryPoints = unitResultsStore.TryGetValue("Product_TrajectoryPoints", out var ptp)
                        ? ptp as List<double[]> : new List<double[]>();
                    var serviceTrajectoryPoints = unitResultsStore.TryGetValue("Service_TrajectoryPoints", out var stp)
                        ? stp as List<double[]> : new List<double[]>();

                    double[] productXPlaneIntersection = unitResultsStore.TryGetValue("Product_XPlaneIntersection", out var pxi)
                        ? pxi as double[] : null;
                    double[] productYPlaneIntersection = unitResultsStore.TryGetValue("Product_YPlaneIntersection", out var pyi)
                        ? pxi as double[] : null;

                    double velocityComponent = (productVelocity + serviceVelocity) / 2.0;
                    double trajectoryStability = 0.5;
                    double intersectionQuality = 0.5;

                    if (productTrajectoryPoints != null && productTrajectoryPoints.Count > 1)
                    {
                        trajectoryStability = CalculateTrajectoryStability(productTrajectoryPoints);
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] QA product trajectory stability: {trajectoryStability:F4}");
                    }

                    if (productXPlaneIntersection != null && productYPlaneIntersection != null)
                    {
                        double zDifference = Math.Abs(productXPlaneIntersection[2] - productYPlaneIntersection[2]);
                        intersectionQuality = 1.0 - Math.Min(1.0, zDifference);
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] QA intersection quality: {intersectionQuality:F4}");
                    }

                    double qaScore = velocityComponent * 0.4 + trajectoryStability * 0.3 + intersectionQuality * 0.3;
                    qaScore = Math.Min(qaScore, 1.0);

                    int qaLevel = (int)(qaScore * 3) + 1;
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] QA final score: {qaScore:F4}, level: {qaLevel}");

                    string result = $"QualityAssessment_Passed_Level_{qaLevel}_V{velocityComponent:F2}_S{trajectoryStability:F2}_I{intersectionQuality:F2}";

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 4 - Feature quality assessment completed: {result}");
                    return result;
                }

                //==========================================================================
                // Step 5: Combined Feature Evaluation
                //==========================================================================
                float Stage5_CombinedFeatureEvaluation(string qualityAssessmentResult, int custId)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 5 - Evaluating combined features for customer {custId}.");

                    double productVelocity = unitResultsStore.TryGetValue("Product_Velocity", out var pv)
                        ? Convert.ToDouble(pv) : 0.5;
                    double serviceVelocity = unitResultsStore.TryGetValue("Service_Velocity", out var sv)
                        ? Convert.ToDouble(sv) : 0.5;

                    int productNegativeBothCount = unitResultsStore.TryGetValue("Product_NegativeBothCount", out var pnbc)
                        ? Convert.ToInt32(pnbc) : 0;
                    int serviceNegativeBothCount = unitResultsStore.TryGetValue("Service_NegativeBothCount", out var snbc)
                        ? Convert.ToInt32(snbc) : 0;

                    int totalNegativePoints = productNegativeBothCount + serviceNegativeBothCount;

                    double[] productVector = unitResultsStore.TryGetValue("Product_Vector", out var pvec)
                        ? pvec as double[] : new double[] { 0, 0, 0 };
                    double[] serviceVector = unitResultsStore.TryGetValue("Service_Vector", out var svec)
                        ? svec as double[] : new double[] { 0, 0, 0 };

                    double alignmentScore = 0.5;
                    double productMagSq = productVector != null ? productVector[0] * productVector[0] + productVector[1] * productVector[1] + productVector[2] * productVector[2] : 0;
                    double serviceMagSq = serviceVector != null ? serviceVector[0] * serviceVector[0] + serviceVector[1] * serviceVector[1] + serviceVector[2] * serviceVector[2] : 0;

                    double productMag = Math.Sqrt(productMagSq);
                    double serviceMag = Math.Sqrt(serviceMagSq);

                    if (productMag > 1e-9 && serviceMag > 1e-9)
                    {
                        double dotProduct = 0;
                        if (productVector != null && serviceVector != null)
                        {
                            for (int i = 0; i < Math.Min(productVector.Length, serviceVector.Length); i++)
                            {
                                dotProduct += productVector[i] * serviceVector[i];
                            }
                        }
                        alignmentScore = dotProduct / (productMag * serviceMag);
                        alignmentScore = Math.Max(-1.0, Math.Min(1.0, alignmentScore));
                        alignmentScore = (alignmentScore + 1.0) / 2.0;
                    }

                    float baseScore = 0.75f + (custId % 10) / 10.0f;
                    float velocityBonus = (float)((productVelocity + serviceVelocity) / 4);
                    float alignmentBonus = (float)(alignmentScore / 5);
                    float negativeBonus = (float)(Math.Min(totalNegativePoints, 10) / 33.33);

                    float result = Math.Min(baseScore + velocityBonus + alignmentBonus + negativeBonus, 1.0f);

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 5 - Combined feature evaluation calculation.");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Base Score: {baseScore:F4}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Velocity Bonus: {velocityBonus:F4} (Product: {productVelocity:F4}, Service: {serviceVelocity:F4})");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Alignment Bonus: {alignmentBonus:F4} (Alignment Score: {alignmentScore:F4})");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Negative Trajectory Bonus: {negativeBonus:F4} (Total Negative Points: {totalNegativePoints})");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Final Score: {result:F4}");

                    return result;
                }

                //==========================================================================
                // Step 6: Fractal Optimization Analysis
                //==========================================================================
                string Stage6_FractalOptimizationAnalysis(string evaluationResult, float evaluationScore, int custId)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 6 - Performing fractal optimization analysis for customer {custId}.");

                    double productVelocity = unitResultsStore.TryGetValue("Product_Velocity", out var pv)
                        ? Convert.ToDouble(pv) : 0.5;
                    double serviceVelocity = unitResultsStore.TryGetValue("Service_Velocity", out var sv)
                        ? Convert.ToDouble(sv) : 0.5;

                    double[] productXPlaneIntersection = unitResultsStore.TryGetValue("Product_XPlaneIntersection", out var pxi)
                        ? pxi as double[] : null;
                    double[] productYPlaneIntersection = unitResultsStore.TryGetValue("Product_YPlaneIntersection", out var pyi)
                        ? pyi as double[] : null;
                    double[] serviceXPlaneIntersection = unitResultsStore.TryGetValue("Service_XPlaneIntersection", out var sxi)
                        ? sxi as double[] : null;
                    double[] serviceYPlaneIntersection = unitResultsStore.TryGetValue("Service_YPlaneIntersection", out var syi)
                        ? syi as double[] : null;

                    Console.WriteLine("========== PRODUCT INTERSECTIONS ==========");
                    if (productXPlaneIntersection != null)
                    {
                        Console.WriteLine($"Product X-Plane Intersection: (0.0, {productXPlaneIntersection[1]:F6}, {productXPlaneIntersection[2]:F6})");
                    }
                    else
                    {
                        Console.WriteLine("Product X-Plane Intersection: null");
                    }

                    if (productYPlaneIntersection != null)
                    {
                        Console.WriteLine($"Product Y-Plane Intersection: ({productYPlaneIntersection[0]:F6}, 0.0, {productYPlaneIntersection[2]:F6})");
                    }
                    else
                    {
                        Console.WriteLine("Product Y-Plane Intersection: null");
                    }

                    Console.WriteLine("========== SERVICE INTERSECTIONS ==========");
                    if (serviceXPlaneIntersection != null)
                    {
                        Console.WriteLine($"Service X-Plane Intersection: (0.0, {serviceXPlaneIntersection[1]:F6}, {serviceXPlaneIntersection[2]:F6})");
                    }
                    else
                    {
                        Console.WriteLine("Service X-Plane Intersection: null");
                    }

                    if (serviceYPlaneIntersection != null)
                    {
                        Console.WriteLine($"Service Y-Plane Intersection: ({serviceYPlaneIntersection[0]:F6}, 0.0, {serviceYPlaneIntersection[2]:F6})");
                    }
                    else
                    {
                        Console.WriteLine("Service Y-Plane Intersection: null");
                    }

                    const int Power = 8;
                    const float EscapeThreshold = 2.0f;
                    const int MaxIterations = 30;

                    float productXPlaneVelocity = (float)(productXPlaneIntersection != null ? productVelocity : 0.0);
                    float productYPlaneVelocity = (float)(productYPlaneIntersection != null ? productVelocity : 0.0);
                    float serviceXPlaneVelocity = (float)(serviceXPlaneIntersection != null ? serviceVelocity : 0.0);
                    float serviceYPlaneVelocity = (float)(serviceYPlaneIntersection != null ? serviceVelocity : 0.0);

                    Console.WriteLine("========== INTERSECTION VELOCITIES ==========");
                    Console.WriteLine($"Product X-Plane Velocity: {productXPlaneVelocity:F4}");
                    Console.WriteLine($"Product Y-Plane Velocity: {productYPlaneVelocity:F4}");
                    Console.WriteLine($"Service X-Plane Velocity: {serviceXPlaneVelocity:F4}");
                    Console.WriteLine($"Service Y-Plane Velocity: {serviceYPlaneVelocity:F4}");

                    List<(System.Numerics.Vector3 position, float velocity, string source)> velocitySources = new List<(System.Numerics.Vector3, float, string)>();

                    if (productXPlaneIntersection != null && productXPlaneIntersection.Length >= 3)
                    {
                        velocitySources.Add((
                            new System.Numerics.Vector3(0.0f, (float)productXPlaneIntersection[1], (float)productXPlaneIntersection[2]),
                            productXPlaneVelocity,
                            "ProductX"));
                    }

                    if (productYPlaneIntersection != null && productYPlaneIntersection.Length >= 3)
                    {
                        velocitySources.Add((
                            new System.Numerics.Vector3((float)productYPlaneIntersection[0], 0.0f, (float)productYPlaneIntersection[2]),
                            productYPlaneVelocity,
                            "ProductY"));
                    }

                    if (serviceXPlaneIntersection != null && serviceXPlaneIntersection.Length >= 3)
                    {
                        velocitySources.Add((
                            new System.Numerics.Vector3(0.0f, (float)serviceXPlaneIntersection[1], (float)serviceXPlaneIntersection[2]),
                            serviceXPlaneVelocity,
                            "ServiceX"));
                    }

                    if (serviceYPlaneIntersection != null && serviceYPlaneIntersection.Length >= 3)
                    {
                        velocitySources.Add((
                            new System.Numerics.Vector3((float)serviceYPlaneIntersection[0], 0.0f, (float)serviceYPlaneIntersection[2]),
                            serviceYPlaneVelocity,
                            "ServiceY"));
                    }

                    Console.WriteLine("========== VELOCITY SOURCES ==========");
                    foreach (var source in velocitySources)
                    {
                        Console.WriteLine($"{source.source} Source Position: ({source.position.X:F4}, {source.position.Y:F4}, {source.position.Z:F4}), Velocity: {source.velocity:F4}");
                    }

                    System.Numerics.Vector3[] samplePoints = new System.Numerics.Vector3[5];

                    samplePoints[0] = (productXPlaneIntersection != null && productXPlaneIntersection.Length >= 3) ?
                        new System.Numerics.Vector3(0.1f, (float)productXPlaneIntersection[1], (float)productXPlaneIntersection[2]) :
                        new System.Numerics.Vector3(0.1f, 0.1f, 0.1f);

                    samplePoints[1] = (productYPlaneIntersection != null && productYPlaneIntersection.Length >= 3) ?
                        new System.Numerics.Vector3((float)productYPlaneIntersection[0], 0.1f, (float)productYPlaneIntersection[2]) :
                        new System.Numerics.Vector3(0.5f, 0.0f, 0.0f);

                    samplePoints[2] = (serviceXPlaneIntersection != null && serviceXPlaneIntersection.Length >= 3) ?
                        new System.Numerics.Vector3(0.1f, (float)serviceXPlaneIntersection[1], (float)serviceXPlaneIntersection[2]) :
                        new System.Numerics.Vector3(0.0f, 0.8f, 0.0f);

                    samplePoints[3] = (serviceYPlaneIntersection != null && serviceYPlaneIntersection.Length >= 3) ?
                        new System.Numerics.Vector3((float)serviceYPlaneIntersection[0], 0.1f, (float)serviceYPlaneIntersection[2]) :
                        new System.Numerics.Vector3(0.3f, 0.3f, 0.3f);

                    if (velocitySources.Count > 0)
                    {
                        System.Numerics.Vector3 sum = System.Numerics.Vector3.Zero;
                        foreach (var source in velocitySources)
                        {
                            sum += source.position;
                        }
                        samplePoints[4] = sum / velocitySources.Count;
                    }
                    else
                    {
                        samplePoints[4] = new System.Numerics.Vector3(1.0f, 1.0f, 1.0f);
                    }

                    Console.WriteLine("========== SAMPLE POINTS ==========");
                    for (int i = 0; i < 5; i++)
                    {
                        Console.WriteLine($"Sample {i + 1} Coordinates: ({samplePoints[i].X:F4}, {samplePoints[i].Y:F4}, {samplePoints[i].Z:F4})");
                    }

                    System.Numerics.Vector3[] sampleValues = new System.Numerics.Vector3[5];
                    int[] sampleIterations = new int[5];
                    float[] sampleVelocities = new float[5];
                    Dictionary<int, Dictionary<string, float>> sampleContributions = new Dictionary<int, Dictionary<string, float>>();

                    for (int i = 0; i < 5; i++)
                    {
                        sampleContributions[i] = new Dictionary<string, float>();
                        foreach (var source in velocitySources)
                        {
                            sampleContributions[i][source.source] = 0.0f;
                        }
                    }

                    for (int sampleIndex = 0; sampleIndex < 5; sampleIndex++)
                    {
                        System.Numerics.Vector3 c = samplePoints[sampleIndex];
                        System.Numerics.Vector3 z = System.Numerics.Vector3.Zero;
                        int iterations = 0;
                        float diffusedVelocity = 0.0f;

                        Console.WriteLine($"========== PROCESSING SAMPLE {sampleIndex + 1} ==========");
                        Console.WriteLine($"Starting point: ({c.X:F4}, {c.Y:F4}, {c.Z:F4})");

                        for (iterations = 0; iterations < MaxIterations; iterations++)
                        {
                            float rSq = z.LengthSquared();

                            if (rSq > EscapeThreshold * EscapeThreshold)
                            {
                                Console.WriteLine($"Escaped at iteration {iterations + 1}");
                                break;
                            }
                            float r = MathF.Sqrt(rSq);

                            Console.WriteLine($"Iteration {iterations + 1}, z=({z.X:F6}, {z.Y:F6}, {z.Z:F6}), r={r:F6}");

                            foreach (var source in velocitySources)
                            {
                                float distanceSq = System.Numerics.Vector3.DistanceSquared(z, source.position);
                                float distance = MathF.Sqrt(distanceSq);

                                if (distance < 2.0f)
                                {
                                    float contribution = source.velocity *
                                                       MathF.Exp(-distance * 2.0f) *
                                                       MathF.Exp(-iterations * 0.1f);

                                    diffusedVelocity += contribution;
                                    if (sampleContributions[sampleIndex].ContainsKey(source.source))
                                    {
                                        sampleContributions[sampleIndex][source.source] += contribution;
                                    }
                                    else
                                    {
                                        sampleContributions[sampleIndex][source.source] = contribution;
                                    }

                                    Console.WriteLine($"  Contribution from {source.source}: {contribution:F6} (distance: {distance:F4})");
                                }
                            }

                            float theta = (r < 1e-6f) ? 0 : MathF.Acos(z.Z / r);
                            float phi = MathF.Atan2(z.Y, z.X);

                            float newR = MathF.Pow(r, Power);
                            float newTheta = Power * theta;
                            float newPhi = Power * phi;

                            z = new System.Numerics.Vector3(
                                newR * MathF.Sin(newTheta) * MathF.Cos(newPhi),
                                newR * MathF.Sin(newTheta) * MathF.Sin(newPhi),
                                newR * MathF.Cos(newTheta)) + c;
                        }

                        sampleValues[sampleIndex] = z;
                        sampleIterations[sampleIndex] = iterations;
                        sampleVelocities[sampleIndex] = diffusedVelocity;

                        Console.WriteLine($"Final Sample {sampleIndex + 1} Results:");
                        Console.WriteLine($"  Final z value: ({z.X:F6}, {z.Y:F6}, {z.Z:F6})");
                        Console.WriteLine($"  Iterations: {iterations}");
                        Console.WriteLine($"  Total diffused velocity: {diffusedVelocity:F6}");
                        Console.WriteLine($"  Contributions breakdown:");
                        foreach (var source in velocitySources)
                        {
                            if (sampleContributions[sampleIndex].ContainsKey(source.source))
                                Console.WriteLine($"    {source.source}: {sampleContributions[sampleIndex][source.source]:F6}");
                        }
                    }

                    unitResultsStore["ProductXPlaneVelocity"] = productXPlaneVelocity;
                    unitResultsStore["ProductYPlaneVelocity"] = productYPlaneVelocity;
                    unitResultsStore["ServiceXPlaneVelocity"] = serviceXPlaneVelocity;
                    unitResultsStore["ServiceYPlaneVelocity"] = serviceYPlaneVelocity;

                    for (int i = 0; i < 5; i++)
                    {
                        unitResultsStore[$"Sample{i + 1}Coordinate"] = samplePoints[i];
                        unitResultsStore[$"Sample{i + 1}Value"] = sampleValues[i];
                        unitResultsStore[$"Sample{i + 1}Iterations"] = sampleIterations[i];
                        unitResultsStore[$"Sample{i + 1}Velocity"] = sampleVelocities[i];

                        foreach (var source in velocitySources)
                        {
                            if (sampleContributions[i].ContainsKey(source.source))
                                unitResultsStore[$"Sample{i + 1}_{source.source}Contribution"] = sampleContributions[i][source.source];
                            else
                                unitResultsStore[$"Sample{i + 1}_{source.source}Contribution"] = 0.0f;
                        }
                    }

                    System.Text.StringBuilder resultBuilder = new System.Text.StringBuilder();
                    resultBuilder.Append($"OptimizationAnalysis_Cust_{custId}");

                    resultBuilder.Append("_V[");
                    bool firstVelocity = true;
                    if (productXPlaneVelocity > 1e-6f || productXPlaneIntersection != null)
                    {
                        resultBuilder.Append($"PX:{productXPlaneVelocity:F3}");
                        firstVelocity = false;
                    }
                    if (productYPlaneVelocity > 1e-6f || productYPlaneIntersection != null)
                    {
                        if (!firstVelocity) resultBuilder.Append(",");
                        resultBuilder.Append($"PY:{productYPlaneVelocity:F3}");
                        firstVelocity = false;
                    }
                    if (serviceXPlaneVelocity > 1e-6f || serviceXPlaneIntersection != null)
                    {
                        if (!firstVelocity) resultBuilder.Append(",");
                        resultBuilder.Append($"SX:{serviceXPlaneVelocity:F3}");
                        firstVelocity = false;
                    }
                    if (serviceYPlaneVelocity > 1e-6f || serviceYPlaneIntersection != null)
                    {
                        if (!firstVelocity) resultBuilder.Append(",");
                        resultBuilder.Append($"SY:{serviceYPlaneVelocity:F3}");
                    }
                    resultBuilder.Append("]");

                    resultBuilder.Append("_S[");
                    for (int i = 0; i < 5; i++)
                    {
                        string status = sampleIterations[i] >= MaxIterations ? "InSet" : $"Escaped({sampleIterations[i]})";
                        resultBuilder.Append($"P{i + 1}:{sampleVelocities[i]:F4}_S{status}");
                        if (i < 4) resultBuilder.Append(",");
                    }
                    resultBuilder.Append("]");

                    string result = resultBuilder.ToString();

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 6 - Fractal optimization analysis completed: {result}");
                    return result;
                }

                //==========================================================================
                // Step 7: Tensor Network Training with Curvature Embedding (Includes Actual TF.NET)
                // PROLIFERATION TRAINING PROCESS:
                // 1. Initial expression "1+P" gets converted to N-dimensional formula
                // 2. During each training epoch, proliferation instance P increments
                // 3. N-dimensional embedding is computed for each batch with current P value
                // 4. Expression influences vertex masks, curvature coefficients, and weight generation
                // 5. Formula proliferates through the network during training iterations
                //==========================================================================
                string Stage7_TensorNetworkTraining(string optimizationResult, int custId, ConcurrentDictionary<string, object> resultsStore)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 7 - Training tensor network for customer {custId} using Actual TF.NET Model A with PROLIFERATION.");

                    tf.compat.v1.disable_eager_execution();
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Disabled eager execution for TensorFlow operations.");

                    byte[]? modelWeightsBytes = RuntimeProcessingContext.RetrieveContextValue("SequentialProcessingUnitC_SerializedModelData") as byte[];
                    byte[]? modelBiasBytes = RuntimeProcessingContext.RetrieveContextValue("SequentialProcessingUnitC_AncillaryData") as byte[];
                    float[] eigenvalues = unitResultsStore.TryGetValue("MarketCurvatureEigenvalues", out var eigVals) && eigVals is float[] eigArray ? eigArray : new float[] { 1.0f, 1.0f, 1.0f };
                    resultsStore["MarketCurvatureEigenvalues"] = eigenvalues;

                    int numEpochs = 100;
                    List<float> trainingLosses = new List<float>();
                    List<float> trainingErrors = new List<float>();

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Step 7 - Creating sample training data.");
                    float[][] numericalSamples = new float[][] {
                new float[] { 0.3f, 0.7f, 0.1f, 0.85f }, new float[] { 0.5f, 0.2f, 0.9f, 0.35f }, new float[] { 0.8f, 0.6f, 0.4f, 0.55f }, new float[] { 0.1f, 0.8f, 0.6f, 0.25f },
                new float[] { 0.7f, 0.3f, 0.2f, 0.95f }, new float[] { 0.4f, 0.5f, 0.7f, 0.65f }, new float[] { 0.2f, 0.9f, 0.3f, 0.15f }, new float[] { 0.6f, 0.1f, 0.8f, 0.75f },
                new float[] { 0.35f, 0.65f, 0.15f, 0.80f }, new float[] { 0.55f, 0.25f, 0.85f, 0.30f }, new float[] { 0.75f, 0.55f, 0.45f, 0.60f }, new float[] { 0.15f, 0.75f, 0.55f, 0.20f },
                new float[] { 0.65f, 0.35f, 0.25f, 0.90f }, new float[] { 0.45f, 0.45f, 0.65f, 0.70f }, new float[] { 0.25f, 0.85f, 0.35f, 0.10f }, new float[] { 0.50f, 0.15f, 0.75f, 0.80f }
            };
                    float[] numericalLabels = new float[numericalSamples.Length];
                    for (int i = 0; i < numericalSamples.Length; i++)
                    {
                        if (numericalSamples[i] == null || numericalSamples[i].Length < 4) { numericalLabels[i] = 0.0f; continue; }
                        float x = numericalSamples[i][0]; float y = numericalSamples[i][1]; float z = numericalSamples[i][2]; float p = numericalSamples[i][3];
                        numericalLabels[i] = x * (float)Math.Cos(p) + y * (float)Math.Sin(p) + z * (float)Math.Cos(p / 2f) + x * y * z * 0.1f;
                    }
                    string[] wordSamples = new string[] {
                "market growth potential high", "customer satisfaction excellent", "product quality superior", "service delivery timely", "price competitiveness average", "brand recognition strong",
                "operational efficiency optimal", "supply chain resilient", "market segment expanding", "customer retention excellent", "product innovation substantial", "service response immediate",
                "price positioning competitive", "brand loyalty increasing", "operational costs decreasing", "supply reliability consistent"
            };
                    float[][] wordEmbeddings = TransformWordsToEmbeddings(wordSamples);
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Created {numericalSamples.Length} numerical samples and {wordSamples.Length} word-based samples.");

                    float[,] numericalData = ConvertJaggedToMultidimensional(numericalSamples);
                    float[,] wordData = ConvertJaggedToMultidimensional(wordEmbeddings);
                    float[,] targetValues = new float[numericalLabels.Length, 1];
                    for (int i = 0; i < numericalLabels.Length; i++) { targetValues[i, 0] = numericalLabels[i]; }

                    int numericalFeatureCount = numericalData.GetLength(1);
                    int wordFeatureCount = wordData.GetLength(1);
                    int totalInputFeatureCount = numericalFeatureCount + wordFeatureCount;

                    int batchSize = 4;
                    int dataSize = numericalData.GetLength(0);
                    int actualBatchSize = Math.Min(batchSize, dataSize);
                    if (actualBatchSize <= 0 && dataSize > 0) actualBatchSize = dataSize;
                    else if (dataSize == 0)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: No training data available. Skipping training.");
                        resultsStore["ModelAProcessingWarning"] = "No training data available."; resultsStore["ModelAProcessingOutcome"] = 0.1f; resultsStore["ModelATrainingError"] = float.NaN;
                        return $"TensorNetworkTrainingSkipped_Cust_{custId}_NoData";
                    }
                    if (actualBatchSize <= 0) actualBatchSize = 1;
                    int numBatches = (actualBatchSize > 0) ? (int)Math.Ceiling((double)dataSize / actualBatchSize) : 0;
                    int[] indices = Enumerable.Range(0, dataSize).ToArray();

                    // === PROLIFERATION INITIALIZATION ===
                    // Start with expression "1+P" where P will proliferate during training
                    string initialExpression = "1+P"; // Changed from "1+1" to "1+P" for proliferation
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] === PROLIFERATION PROCESS BEGINS ===");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Initial expression: {initialExpression}");

                    if (modelWeightsBytes != null && modelBiasBytes != null && modelWeightsBytes.Length > 0 && modelBiasBytes.Length > 0)
                    {
                        modelAGraph = tf.Graph();
                        modelAGraph.as_default();
                        {
                            modelASession = tf.Session(modelAGraph);
                            try
                            {
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Step 7 - Initializing Model A Architecture in its own graph.");
                                float[] unitCWeightsArray = DeserializeFloatArray(modelWeightsBytes);
                                float[] unitCBiasArray = DeserializeFloatArray(modelBiasBytes);
                                int unitCHiddenSize = -1;
                                if (unitCBiasArray.Length >= 1) unitCHiddenSize = unitCBiasArray.Length - 1;
                                if (unitCHiddenSize <= 0 || unitCWeightsArray.Length != (totalInputFeatureCount * unitCHiddenSize) + (unitCHiddenSize * 1))
                                {
                                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: Could not reliably infer Unit C hidden size. Using fallback.");
                                    unitCHiddenSize = 64;
                                }
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Model A architecture parameters: Input Feats: {totalInputFeatureCount}, Hidden Size: {unitCHiddenSize}");

                                // Initial proliferation instance starts at 1
                                int initialProliferationInstance = 1;
                                string regexPattern = ConvertExpressionToRegex(initialExpression);
                                string nDimensionalExpression = ConvertRegexToNDimensionalExpression(regexPattern, initialProliferationInstance);

                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] STEP 1→2: Expression '{initialExpression}' → Regex '{regexPattern}' → N-Dim '{nDimensionalExpression}'");

                                // Generate initial weights with proliferation
                                float[,] modelAWeights1Data = GenerateWeightsFromExpression(nDimensionalExpression, totalInputFeatureCount, unitCHiddenSize, initialProliferationInstance);
                                float[,] modelAWeights2Data = GenerateWeightsFromExpression(nDimensionalExpression, unitCHiddenSize, 1, initialProliferationInstance);

                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Defining TensorFlow operations for Model A with proliferation support.");
                                Tensor numericalInput = tf.placeholder(tf.float32, shape: new int[] { -1, numericalFeatureCount }, name: "numerical_input_A");
                                Tensor wordInput = tf.placeholder(tf.float32, shape: new int[] { -1, wordFeatureCount }, name: "word_input_A");
                                Tensor targetOutput = tf.placeholder(tf.float32, shape: new int[] { -1, 1 }, name: "target_output_A");
                                Tensor combinedInput = tf.concat(new[] { numericalInput, wordInput }, axis: 1, name: "combined_input_A");

                                // Add N-dimensional embedding input for proliferation
                                Tensor ndEmbeddingInput = tf.placeholder(tf.float32, shape: new int[] { -1, 16 }, name: "nd_embedding_input_A");
                                Tensor extendedInput = tf.concat(new[] { combinedInput, ndEmbeddingInput }, axis: 1, name: "extended_input_A");

                                // Adjust weight dimensions for extended input
                                int extendedInputSize = totalInputFeatureCount + 16; // Add 16 for N-dimensional embedding
                                float[,] extendedWeights1Data = GenerateWeightsFromExpression(nDimensionalExpression, extendedInputSize, unitCHiddenSize, initialProliferationInstance);

                                ResourceVariable weights1 = tf.Variable(tf.constant(extendedWeights1Data, dtype: tf.float32), name: "weights1_A");
                                ResourceVariable bias1 = tf.Variable(tf.zeros(unitCHiddenSize, dtype: tf.float32), name: "bias1_A");
                                ResourceVariable weights2 = tf.Variable(tf.constant(modelAWeights2Data, dtype: tf.float32), name: "weights2_A");
                                ResourceVariable bias2 = tf.Variable(tf.zeros(1, dtype: tf.float32), name: "bias2_A");

                                Tensor hidden = tf.nn.relu(tf.add(tf.matmul(extendedInput, weights1), bias1), name: "hidden_A");

                                // STEP 6: Apply vertex mask with proliferation to hidden layer
                                Tensor vertexMask = CalculateOutermostVertexMask(hidden, initialProliferationInstance);
                                Tensor maskedHidden = tf.multiply(hidden, vertexMask, name: "masked_hidden_A");

                                Tensor predictions = tf.add(tf.matmul(maskedHidden, weights2), bias2, name: "predictions_A");
                                Tensor loss = tf.reduce_mean(tf.square(tf.subtract(predictions, targetOutput)), name: "mse_loss_A");
                                var optimizer = tf.train.AdamOptimizer(0.001f);
                                Operation trainOp = optimizer.minimize(loss);
                                Tensor meanAbsError = tf.reduce_mean(tf.abs(tf.subtract(predictions, targetOutput)), name: "mae_A");
                                Operation initOp = tf.global_variables_initializer();
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] TensorFlow operations defined for Model A with proliferation.");

                                modelASession.run(initOp);
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Model A - Actual TensorFlow.NET variables initialized in its own session.");

                                // === PROLIFERATION TRAINING LOOP ===
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] === BEGINNING PROLIFERATION TRAINING ===");

                                for (int epoch = 0; epoch < numEpochs; epoch++)
                                {
                                    // PROLIFERATION: Calculate current proliferation instance
                                    // P starts at 1 and increases every 10 epochs
                                    int currentProliferationInstance = 1 + (epoch / 10);

                                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] === EPOCH {epoch + 1}/{numEpochs} - PROLIFERATION INSTANCE P={currentProliferationInstance} ===");

                                    // STEP 2: Generate new N-dimensional expression for current proliferation
                                    string currentRegexPattern = ConvertExpressionToRegex(initialExpression);
                                    string currentNDExpression = ConvertRegexToNDimensionalExpression(currentRegexPattern, currentProliferationInstance);
                                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Current N-Dimensional Expression: {currentNDExpression}");

                                    ShuffleArray(indices);
                                    float epochLoss = 0.0f;

                                    for (int batch = 0; batch < numBatches; batch++)
                                    {
                                        int startIdx = batch * actualBatchSize;
                                        int endIdx = Math.Min(startIdx + actualBatchSize, dataSize);
                                        int batchCount = endIdx - startIdx;
                                        if (batchCount <= 0) continue;

                                        float[,] batchNumerical = ExtractBatch(numericalData, indices, startIdx, batchCount);
                                        float[,] batchWord = ExtractBatch(wordData, indices, startIdx, batchCount);
                                        float[,] batchTarget = ExtractBatch(targetValues, indices, startIdx, batchCount);

                                        // STEP 3: Compute N-dimensional embedding for current batch with current proliferation
                                        float[,] batchCombined = new float[batchCount, numericalFeatureCount + wordFeatureCount];
                                        for (int i = 0; i < batchCount; i++)
                                        {
                                            for (int j = 0; j < numericalFeatureCount; j++)
                                            {
                                                batchCombined[i, j] = batchNumerical[i, j];
                                            }
                                            for (int j = 0; j < wordFeatureCount; j++)
                                            {
                                                batchCombined[i, numericalFeatureCount + j] = batchWord[i, j];
                                            }
                                        }

                                        // PROLIFERATION: Compute N-dimensional embedding with current P value
                                        float[,] ndEmbedding = ComputeNDimensionalEmbedding(batchCombined, currentProliferationInstance, 16);

                                        var batchFeed = new FeedItem[] {
                                    new FeedItem(numericalInput, batchNumerical),
                                    new FeedItem(wordInput, batchWord),
                                    new FeedItem(ndEmbeddingInput, ndEmbedding), // Include N-dimensional embedding
                                    new FeedItem(targetOutput, batchTarget)
                                };

                                        var results = modelASession.run(new ITensorOrOperation[] { loss, trainOp }, batchFeed);
                                        float batchLossValue = ((Tensor)results[0]).numpy().ToArray<float>()[0];
                                        epochLoss += batchLossValue;

                                        if (batch % 5 == 0 || batch == numBatches - 1)
                                        {
                                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Epoch {epoch + 1}/{numEpochs}, Batch {batch + 1}/{numBatches}, P={currentProliferationInstance}, Batch Loss: {batchLossValue:F6}");
                                        }
                                    }

                                    if (numBatches > 0) epochLoss /= numBatches;
                                    else epochLoss = float.NaN;
                                    trainingLosses.Add(epochLoss);

                                    if (epoch % 10 == 0 || epoch == numEpochs - 1)
                                    {
                                        // Evaluate with current proliferation instance
                                        float[,] fullCombined = new float[dataSize, numericalFeatureCount + wordFeatureCount];
                                        for (int i = 0; i < dataSize; i++)
                                        {
                                            for (int j = 0; j < numericalFeatureCount; j++)
                                            {
                                                fullCombined[i, j] = numericalData[i, j];
                                            }
                                            for (int j = 0; j < wordFeatureCount; j++)
                                            {
                                                fullCombined[i, numericalFeatureCount + j] = wordData[i, j];
                                            }
                                        }

                                        float[,] evalNdEmbedding = ComputeNDimensionalEmbedding(fullCombined, currentProliferationInstance, 16);

                                        var trainingDataFeed = new FeedItem[] {
                                    new FeedItem(numericalInput, numericalData),
                                    new FeedItem(wordInput, wordData),
                                    new FeedItem(ndEmbeddingInput, evalNdEmbedding),
                                    new FeedItem(targetOutput, targetValues)
                                };

                                        var evalResults = modelASession.run(new ITensorOrOperation[] { meanAbsError }, trainingDataFeed);
                                        float currentErrorValue = ((Tensor)evalResults[0]).numpy().ToArray<float>()[0];
                                        trainingErrors.Add(currentErrorValue);
                                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Epoch {epoch + 1}/{numEpochs}, P={currentProliferationInstance}, Average Loss: {(float.IsNaN(epochLoss) ? "N/A" : epochLoss.ToString("F6"))}, Mean Absolute Error: {currentErrorValue:F6}");
                                    }
                                }

                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] === PROLIFERATION TRAINING COMPLETED ===");
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Model A training completed - Expression '1+P' proliferated through {numEpochs} epochs");

                                // Final evaluation with the last proliferation instance
                                int finalProliferationInstance = 1 + ((numEpochs - 1) / 10);
                                float[,] finalCombined = new float[dataSize, numericalFeatureCount + wordFeatureCount];
                                for (int i = 0; i < dataSize; i++)
                                {
                                    for (int j = 0; j < numericalFeatureCount; j++)
                                    {
                                        finalCombined[i, j] = numericalData[i, j];
                                    }
                                    for (int j = 0; j < wordFeatureCount; j++)
                                    {
                                        finalCombined[i, numericalFeatureCount + j] = wordData[i, j];
                                    }
                                }

                                float[,] finalNdEmbedding = ComputeNDimensionalEmbedding(finalCombined, finalProliferationInstance, 16);

                                var finalTrainingDataFeed = new FeedItem[] {
                            new FeedItem(numericalInput, numericalData),
                            new FeedItem(wordInput, wordData),
                            new FeedItem(ndEmbeddingInput, finalNdEmbedding),
                            new FeedItem(targetOutput, targetValues)
                        };

                                var finalResults = modelASession.run(new ITensorOrOperation[] { meanAbsError, predictions }, finalTrainingDataFeed);
                                float finalErrorValue = ((Tensor)finalResults[0]).numpy().ToArray<float>()[0];
                                Tensor finalPredictionsTensor = (Tensor)finalResults[1];
                                float[] finalPredictionsFlat = finalPredictionsTensor.ToArray<float>();
                                int[] finalPredictionsDims = finalPredictionsTensor.shape.dims.Select(d => (int)d).ToArray();

                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Model A Final Predictions Shape: {string.Join(",", finalPredictionsDims)}");
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Model A Final Predictions (First few): [{string.Join(", ", finalPredictionsFlat.Take(Math.Min(finalPredictionsFlat.Length, 10)).Select(p => p.ToString("F4")))}...]");

                                var finalParams = modelASession.run(new ITensorOrOperation[] { weights1.AsTensor(), bias1.AsTensor(), weights2.AsTensor(), bias2.AsTensor() });
                                var finalWeights1 = ((Tensor)finalParams[0]).ToArray<float>();
                                var finalBias1 = ((Tensor)finalParams[1]).ToArray<float>();
                                var finalWeights2 = ((Tensor)finalParams[2]).ToArray<float>();
                                var finalBias2 = ((Tensor)finalParams[3]).ToArray<float>();

                                byte[] trainedWeights1Bytes = SerializeFloatArray(finalWeights1);
                                byte[] trainedBias1Bytes = SerializeFloatArray(finalBias1);
                                byte[] trainedWeights2Bytes = SerializeFloatArray(finalWeights2);
                                byte[] trainedBias2Bytes = SerializeFloatArray(finalBias2);

                                var byteArraysToCombine = new List<byte[]>();
                                if (trainedWeights1Bytes != null) byteArraysToCombine.Add(trainedWeights1Bytes);
                                if (trainedBias1Bytes != null) byteArraysToCombine.Add(trainedBias1Bytes);
                                if (trainedWeights2Bytes != null) byteArraysToCombine.Add(trainedWeights2Bytes);
                                if (trainedBias2Bytes != null) byteArraysToCombine.Add(trainedBias2Bytes);
                                byte[] combinedModelAData = byteArraysToCombine.SelectMany(arr => arr).ToArray();

                                if (finalPredictionsFlat != null && finalPredictionsFlat.Length > 0)
                                {
                                    resultsStore["ModelAPredictionsFlat"] = finalPredictionsFlat;
                                    resultsStore["ModelAPredictionsShape"] = finalPredictionsDims;
                                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Model A Predictions stored in results dictionary ({finalPredictionsFlat.Length} values)");
                                }
                                else
                                {
                                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] WARNING: Model A failed to produce predictions to store");
                                }

                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Model A Final Mean Absolute Error: {finalErrorValue:F6}");
                                float modelAOutcomeScore = Math.Max(0.0f, 1.0f - finalErrorValue / 0.5f);
                                resultsStore["ModelAProcessingOutcome"] = modelAOutcomeScore;
                                resultsStore["ModelATrainingError"] = finalErrorValue;
                                resultsStore["ModelATrainingLosses"] = trainingLosses.ToArray();
                                resultsStore["ModelATrainingErrors"] = trainingErrors.ToArray();
                                resultsStore["ModelACombinedParameters"] = combinedModelAData;
                                resultsStore["ModelAFinalProliferationInstance"] = finalProliferationInstance;
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Model A results dictionary populated with {resultsStore.Count} entries");

                                var modelMetadata = new Dictionary<string, object> {
                            { "EmbeddedExpression", initialExpression },
                            { "NDimensionalExpression", nDimensionalExpression },
                            { "TrainingEpochs", numEpochs },
                            { "FinalMeanAbsoluteError", finalErrorValue },
                            { "TotalInputFeatureCount", totalInputFeatureCount },
                            { "HiddenLayerSize", unitCHiddenSize },
                            { "TrainingSampleCount", dataSize },
                            { "CreationTimestamp", DateTime.UtcNow.ToString("o") },
                            { "CurvatureEigenvalues", eigenvalues },
                            { "HasOutermostVertexFocus", true },
                            { "UsesNDimensionalIterations", true },
                            { "FinalProliferationInstance", finalProliferationInstance },
                            { "ProliferationEnabled", true }
                        };
                                string metadataJson = SerializeMetadata(modelMetadata);
                                byte[] metadataBytes = System.Text.Encoding.UTF8.GetBytes(metadataJson);
                                resultsStore["ModelAMetadata"] = metadataBytes;
                                RuntimeProcessingContext.StoreContextValue("model_a_params_combined", combinedModelAData);
                                RuntimeProcessingContext.StoreContextValue("model_a_metadata", metadataBytes);
                                RuntimeProcessingContext.StoreContextValue("model_a_expression", initialExpression);
                                RuntimeProcessingContext.StoreContextValue("model_a_expression_nd", nDimensionalExpression);
                                RuntimeProcessingContext.StoreContextValue("model_a_final_proliferation", finalProliferationInstance);
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Step 7 - Model A parameters saved to RuntimeContext (Size: {combinedModelAData?.Length ?? 0} bytes)");
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Step 7 - Model A trained and saved to RuntimeProcessingContext and Results Store with final proliferation instance: {finalProliferationInstance}");

                                string result = $"TensorNetworkTrained_Cust_{custId}_MAE{finalErrorValue:F4}_Expr({initialExpression.Replace('+', 'p')})_FinalP{finalProliferationInstance}";
                                return result;
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Error during Step 7 - Tensor Network Training (Model A Graph Context): {ex.Message}");
                                resultsStore["ModelAProcessingError"] = "Model A Training Error: " + ex.Message;
                                resultsStore["ModelAProcessingOutcome"] = 0.0f;
                                resultsStore["ModelATrainingError"] = float.NaN;
                                throw new InvalidOperationException($"Model A training failed: {ex.Message}", ex);
                            }
                        }
                    }
                    else
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Step 7 - Missing initial model parameters from Unit C for Model A training. Skipping training.");
                        resultsStore["ModelAProcessingWarning"] = "Missing initial parameters from Unit C for training.";
                        resultsStore["ModelAProcessingOutcome"] = 0.1f;
                        resultsStore["ModelATrainingError"] = float.NaN;
                        return $"TensorNetworkTrainingSkipped_Cust_{custId}_MissingData";
                    }
                }

                //==========================================================================
                // Step 8: Future Performance Projection
                //==========================================================================
                string Stage8_FutureProjection(string trainingOutcomeResult, float evaluationScore, int custId)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 8 - Generating future performance projection for customer {custId}.");

                    float combinedFeatureEvaluationScore = unitResultsStore.TryGetValue("CombinedEvaluationScore", out var evalScore)
                        ? Convert.ToSingle(evalScore) : evaluationScore;

                    float modelATrainingOutcomeScore = unitResultsStore.TryGetValue("ModelAProcessingOutcome", out var maeScore)
                        ? Convert.ToSingle(maeScore) : 0.0f;

                    float modelATrainingError = unitResultsStore.TryGetValue("ModelATrainingError", out var maeError)
                        && maeError is float maeFloat ? maeFloat : float.NaN;

                    byte[]? modelACombinedParams = unitResultsStore.TryGetValue("ModelACombinedParameters", out var maParams)
                        ? maParams as byte[] : null;

                    int finalProliferationInstance = unitResultsStore.TryGetValue("ModelAFinalProliferationInstance", out var fpi)
                        ? Convert.ToInt32(fpi) : 1;

                    string projectionOutcome = "Stable";
                    float projectedScore = (combinedFeatureEvaluationScore + modelATrainingOutcomeScore) / 2.0f;

                    // Factor in proliferation instance for projection
                    float proliferationBonus = (finalProliferationInstance - 1) * 0.02f; // Small bonus for higher proliferation
                    projectedScore = Math.Min(projectedScore + proliferationBonus, 1.0f);

                    if (!float.IsNaN(modelATrainingError))
                    {
                        if (modelATrainingError < 0.05f)
                        {
                            projectionOutcome = "StrongGrowth_HighProliferation";
                            projectedScore = Math.Min(projectedScore + 0.1f, 1.0f);
                        }
                        else if (modelATrainingError > 0.2f)
                        {
                            projectionOutcome = "PotentialChallenges";
                            projectedScore = Math.Max(projectedScore - 0.05f, 0.0f);
                        }
                    }
                    else
                    {
                        projectionOutcome = "TrainingDataUnavailable";
                    }

                    if (modelACombinedParams != null && modelACombinedParams.Length > 1000)
                    {
                        projectionOutcome += "_ComplexModel";
                        if (!float.IsNaN(modelATrainingError))
                        {
                            if (modelATrainingError < 0.1f)
                            {
                                projectedScore = Math.Min(projectedScore + 0.03f, 1.0f);
                            }
                            else if (modelATrainingError > 0.3f)
                            {
                                projectedScore = Math.Max(projectedScore - 0.03f, 0.0f);
                            }
                        }
                    }

                    projectedScore = Math.Max(0.0f, Math.Min(1.0f, projectedScore));
                    unitResultsStore["ProjectedPerformanceScore"] = projectedScore;

                    string result = $"PerformanceProjection_Cust_{custId}_Outcome_{projectionOutcome}_Score_{projectedScore:F4}_TrainError_{(float.IsNaN(modelATrainingError) ? "N/A" : modelATrainingError.ToString("F4"))}_FinalP{finalProliferationInstance}";

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 8 - Future performance projection completed: {result}");
                    return result;
                }

                #endregion

                //==========================================================================
                // Workflow Execution
                //==========================================================================
                var workflowResult = ExecuteProductionWorkflow(outcomeRecord, customerIdentifier, mlSession_param_unused);
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Workflow completed with result: {workflowResult}");

                await Task.Delay(250);
            }
            catch (Exception ex)
            {
                //==========================================================================
                // Error Handling
                //==========================================================================
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Error in Parallel Processing Unit A: {ex.Message}");
                unitResultsStore["ModelAProcessingError"] = ex.Message;
                unitResultsStore["ModelAProcessingOutcome"] = 0.0f;
                throw;
            }
            finally
            {
                //==========================================================================
                // Cleanup
                //==========================================================================
                MlProcessOrchestrator.DisposeGraphAndSession(ref modelAGraph, ref modelASession);
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Parallel Processing Unit A TF Graph and Session disposed.");
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Parallel Processing Unit A finished.");
            }
        }























        /// <summary>
        /// Processes data simulating Model B (ParallelProcessingUnitB).
        /// This method is designed to run in parallel with ParallelProcessingUnitA (Actual Model A).
        /// It performs another actual TensorFlow.NET operation based on the core outcome record data
        /// (potentially using the model data generated by Unit C) and stores its results in a shared thread-safe dictionary.
        /// </summary>
        /// <param name="outcomeRecord">The core CoreMlOutcomeRecord object established by SequentialInitialProcessingUnitC.</param>
        /// <param name="customerIdentifier">The customer identifier.</param>
        /// <param name="requestSequenceIdentifier">The request session identifier.</param>
        /// <param name="mlSession_param_unused">A dedicated actual TensorFlow.NET Session environment for this parallel task (now largely unused internally by this unit's TF ops).</param>
        /// <param name="unitResultsStore">A thread-safe dictionary to store results for SequentialFinalProcessingUnitD.</param>
        private async Task ParallelProcessingUnitB(CoreMlOutcomeRecord outcomeRecord, int customerIdentifier, int requestSequenceIdentifier, Tensorflow.Session mlSession_param_unused, ConcurrentDictionary<string, object> unitResultsStore)
        {
            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Starting Parallel Processing Unit B for customer {customerIdentifier}.");
            Tensorflow.Graph modelBGraph = null;
            Tensorflow.Session modelBSession = null;

            try
            {
                //==========================================================================
                // General Utility Methods (Accessible by all Stages within this Unit)
                // These methods are defined *inside* this ParallelProcessingUnitB method
                // to match the original structure and resolve scope issues.
                //==========================================================================

                #region Helper Methods (Required by this method for Unit B)

                /// <summary>
                /// Converts a simple mathematical expression into a regular expression pattern.
                /// </summary>
                string ConvertExpressionToRegex(string expression)
                {
                    // For the simple expression "2*2", create a regex pattern with capture groups
                    if (expression == "2*2")
                    {
                        return @"(\d+)([\+\-\*\/])(\d+)";
                    }

                    // For more complex expressions, implement a more sophisticated parser
                    string pattern = expression.Replace("2", @"(\d+)");
                    pattern = pattern.Replace("*", @"([\+\-\*\/])");

                    return pattern;
                }

                /// <summary>
                /// Converts a regular expression pattern into an n-dimensional compute-safe expression.
                /// </summary>
                string ConvertRegexToNDimensionalExpression(string regexPattern)
                {
                    // Convert the pattern to an n-dimensional expression
                    if (regexPattern == @"(\d+)([\+\-\*\/])(\d+)" || (regexPattern?.Contains(@"(\d+)") == true && regexPattern?.Contains(@"([\+\-\*\/])") == true)) // Added null check and boolean logic
                    {
                        // For the "2*2" -> 4 result, we want an expression that might amplify or interact
                        // Let's map this to ND(x,y,z,p)=Vx*sin(p)+Vy*cos(p)+Vz*sin(p/2)
                        // This is similar to Unit A but with sin/cos swapped on X/Y, and the "2*2" result
                        // influencing amplitude or frequency, which we'll apply during weight generation.
                        return "ND(x,y,z,p)=Vx*sin(p)+Vy*cos(p)+Vz*sin(p/2)";
                    }

                    // Default fallback for unrecognized patterns
                    return "ND(x,y,z,p)=x+y+z";
                }

                /// <summary>
                /// Transforms word-based samples into numerical embeddings using a simplified embedding technique.
                /// </summary>
                float[][] TransformWordsToEmbeddings(string[] wordSamples)
                {
                    if (wordSamples == null) return new float[0][]; // Added null check
                    // Simple word embedding function - in a real implementation, this would use
                    // proper word embeddings from a pre-trained model or custom embeddings

                    // For this example, we'll convert each word sample to a 10-dimensional vector
                    // using a consistent but simplified approach
                    int embeddingDimensions = 10;
                    float[][] embeddings = new float[wordSamples.Length][];

                    for (int i = 0; i < wordSamples.Length; i++)
                    {
                        embeddings[i] = new float[embeddingDimensions];
                        if (wordSamples[i] == null) continue; // Skip null samples


                        // Split the sample into words
                        string[] words = wordSamples[i]?.Split(' ') ?? new string[0]; // Handle null sample

                        // For each word, contribute to the embedding
                        for (int j = 0; j < words.Length; j++)
                        {
                            string word = words[j];
                            if (string.IsNullOrEmpty(word)) continue; // Skip empty words

                            // Use a simple hash function to generate values from words
                            // Ensure the hash function is consistent and spreads values
                            int hashBase = word.GetHashCode();
                            for (int k = 0; k < embeddingDimensions; k++)
                            {
                                // Generate a value based on hash, dimension, and word index
                                // Using a prime multiplier to help distribute values
                                int valueInt = Math.Abs(hashBase * (k + 1) * (j + 1) * 31); // Multiply by prime 31
                                float value = (valueInt % 1000) / 1000.0f; // Map to 0-1 range

                                // Add to the embedding with position-based weighting (inverse word index)
                                embeddings[i][k] += value * (1.0f / (j + 1.0f)); // Use 1.0f for float division
                            }
                        }

                        // Normalize the embedding vector
                        float magnitudeSq = 0;
                        for (int k = 0; k < embeddingDimensions; k++)
                        {
                            magnitudeSq += embeddings[i][k] * embeddings[i][k];
                        }

                        float magnitude = (float)Math.Sqrt(magnitudeSq);
                        if (magnitude > 1e-6f) // Use tolerance
                        {
                            for (int k = 0; k < embeddingDimensions; k++)
                            {
                                embeddings[i][k] /= magnitude;
                            }
                        }
                    }

                    return embeddings;
                }

                /// <summary>
                /// Applies the n-dimensional expression to curvature coefficients.
                /// </summary>
                float[] ApplyNDimensionalExpressionToCurvature(float[] coefficients, string ndExpression)
                {
                    if (coefficients == null) return new float[0]; // Added null check


                    // Parse the expression to determine how to modify coefficients
                    // For our example derived from "2*2", we'll implement the
                    // equivalent of "multiplying by 4" to appropriate coefficients

                    float[] modifiedCoefficients = new float[coefficients.Length];
                    System.Buffer.BlockCopy(coefficients, 0, modifiedCoefficients, 0, coefficients.Length * sizeof(float)); // Use System.Buffer

                    // Apply the expression's effect to the coefficients
                    if (ndExpression.StartsWith("ND(x,y,z,p)="))
                    {
                        // Extract the dimensional scaling factors from the expression
                        // For ND(x,y,z,p)=Vx*sin(p)+Vy*cos(p)+Vz*sin(p/2), the 'V' factors are implicitly 1 in this simplified mapping.
                        // The "2*2=4" idea suggests a scaling or multiplicative influence.
                        // Let's map the "2*2" effect to scale the primary diagonal coefficients (xx, yy, zz) by 4.
                        float scalingFactor = 4.0f; // Represents the "2*2" result influencing scale

                        if (modifiedCoefficients.Length > 0) modifiedCoefficients[0] *= scalingFactor;  // xx coefficient
                        if (modifiedCoefficients.Length > 1) modifiedCoefficients[1] *= scalingFactor;  // yy coefficient
                        if (modifiedCoefficients.Length > 2) modifiedCoefficients[2] *= scalingFactor;  // zz coefficient

                        // Scale cross-terms relative to the primary terms' scaling
                        float crossTermScale = (scalingFactor + scalingFactor) / 2.0f; // Simple average influence
                        if (modifiedCoefficients.Length > 3) modifiedCoefficients[3] *= crossTermScale;  // xy coefficient
                        if (modifiedCoefficients.Length > 4) modifiedCoefficients[4] *= crossTermScale;  // xz coefficient
                        if (modifiedCoefficients.Length > 5) modifiedCoefficients[5] *= crossTermScale;  // yz coefficient

                        // Apply a lesser influence to higher-order terms if they exist
                        float higherOrderScale = (crossTermScale + 1.0f) / 2.0f; // Average of cross-term scale and default 1.0
                        if (modifiedCoefficients.Length > 6) modifiedCoefficients[6] *= higherOrderScale;
                        if (modifiedCoefficients.Length > 7) modifiedCoefficients[7] *= higherOrderScale;
                        if (modifiedCoefficients.Length > 8) modifiedCoefficients[8] *= higherOrderScale;
                    }

                    return modifiedCoefficients;
                }


                /// <summary>
                /// Generates weight matrices from our n-dimensional expression.
                /// </summary>
                float[,] GenerateWeightsFromExpression(string expression, int inputDim, int outputDim)
                {
                    if (inputDim <= 0 || outputDim <= 0)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper (Unit B): GenerateWeightsFromExpression received invalid dimensions ({inputDim}, {outputDim}). Returning empty array.");
                        return new float[0, 0];
                    }
                    // Create a weight matrix based on our expression
                    float[,] weights = new float[inputDim, outputDim];

                    // Use a pseudorandom generator with fixed seed for reproducibility
                    // Use a different seed base than Unit A
                    Random rand = new Random(43 + inputDim * 100 + outputDim); // Vary seed slightly for different matrix sizes

                    // Fill the matrix with values derived from our expression
                    for (int i = 0; i < inputDim; i++)
                    {
                        for (int j = 0; j < outputDim; j++)
                        {
                            // Base weight value (small random initialization)
                            float baseWeight = (float)(rand.NextDouble() * 0.04 - 0.02); // Range -0.02 to +0.02

                            // Apply influence from our expression's structure
                            // For "2*2" -> ND(x,y,z,p)=Vx*sin(p)+Vy*cos(p)+Vz*sin(p/2)
                            // We want the weights to reflect the oscillatory and dimensional coupling nature,
                            // with potentially different phase shifts or scales compared to Unit A.
                            float expressionInfluence = (float)(
                                Math.Sin((i + j) * Math.PI / (inputDim + outputDim) * 1.5) + // Sine based on combined indices (different frequency)
                                Math.Cos(i * Math.PI / inputDim) * 0.6 +               // Cosine based on input index (different scale)
                                Math.Sin(j * Math.PI / (outputDim * 1.5)) * 0.4         // Sine based on output index (different frequency and scale)
                            );


                            // Scale the influence and add it to the base weight
                            // The scale factor ensures the expression doesn't dominate initialization completely
                            float influenceScale = 0.12f; // Slightly different influence scale than Unit A
                            weights[i, j] = baseWeight + expressionInfluence * influenceScale;
                        }
                    }

                    // Enhance the "outermost vertices" (corners) to emphasize the boundary condition - Different boost than Unit A
                    float cornerBoost = 1.7f; // Factor to multiply corner weights by
                    if (inputDim > 0 && outputDim > 0)
                    {
                        weights[0, 0] *= cornerBoost;                   // Top-left
                        weights[0, outputDim - 1] *= cornerBoost;        // Top-right
                        weights[inputDim - 1, 0] *= cornerBoost;         // Bottom-left
                        weights[inputDim - 1, outputDim - 1] *= cornerBoost; // Bottom-right (outermost conceptual vertex)
                    }

                    return weights;
                }

                /// <summary>
                /// Calculates a basis vector from sample coordinates along a specific dimension.
                /// </summary>
                System.Numerics.Vector3 CalculateBasisVector(System.Numerics.Vector3[] coordinates, int dimension)
                {
                    if (coordinates == null || coordinates.Length == 0 || dimension < 0 || dimension > 2) // Added null/empty/dimension checks
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: Invalid input for CalculateBasisVector. Returning zero vector.");
                        return System.Numerics.Vector3.Zero;
                    }

                    // Start with a zero vector
                    System.Numerics.Vector3 basis = System.Numerics.Vector3.Zero;

                    // For each coordinate, extract the component from the specified dimension
                    // and use it to weight the contribution of that coordinate to the basis vector
                    foreach (var coord in coordinates)
                    {
                        // System.Numerics.Vector3 is a struct, so no null check needed for 'coord' itself after array check
                        // Get the component value for the specified dimension
                        float component = dimension == 0 ? coord.X : (dimension == 1 ? coord.Y : coord.Z);

                        // Add the weighted coordinate to the basis vector
                        basis += new System.Numerics.Vector3(
                            coord.X * component,
                            coord.Y * component,
                            coord.Z * component
                        );
                    }

                    // Normalize the basis vector to unit length
                    float magnitude = basis.Length();
                    if (magnitude > 1e-6f) // Use tolerance
                    {
                        basis = System.Numerics.Vector3.Divide(basis, magnitude);
                    }

                    return basis;
                }

                /// <summary>
                /// Calculates coefficients that represent how the curvature varies in the sample space.
                /// </summary>
                float[] CalculateCurvatureCoefficients(System.Numerics.Vector3[] coordinates, System.Numerics.Vector3[] values)
                {
                    if (coordinates == null || values == null || coordinates.Length == 0 || coordinates.Length != values.Length)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: Invalid input for CalculateCurvatureCoefficients. Returning zero coefficients.");
                        return new float[9]; // Return zero array if input is invalid
                    }


                    // We'll use 9 coefficients to represent the curvature in 3D space
                    float[] coefficients = new float[9];

                    // For each coordinate-value pair
                    for (int i = 0; i < coordinates.Length; i++)
                    {
                        System.Numerics.Vector3 coord = coordinates[i];
                        System.Numerics.Vector3 value = values[i];

                        // Calculate the squared components of the coordinate
                        float x2 = coord.X * coord.X;
                        float y2 = coord.Y * coord.Y;
                        float z2 = coord.Z * coord.Z;

                        // Calculate the cross-components
                        float xy = coord.X * coord.Y;
                        float xz = coord.X * coord.Z;
                        float yz = coord.Y * coord.Z;

                        // Calculate the dot product of coordinate and value
                        float dot = System.Numerics.Vector3.Dot(coord, value);

                        // Update the coefficients based on this sample
                        coefficients[0] += x2 * dot; // xx component
                        coefficients[1] += y2 * dot; // yy component
                        coefficients[2] += z2 * dot; // zz component
                        coefficients[3] += xy * dot; // xy component
                        coefficients[4] += xz * dot; // xz component
                        coefficients[5] += yz * dot; // yz component
                        coefficients[6] += x2 * y2 * dot; // xxyy component (higher order)
                        coefficients[7] += x2 * z2 * dot; // xxzz component (higher order)
                        coefficients[8] += y2 * z2 * dot; // yyzz component (higher order)
                    }

                    // Normalize the coefficients by the number of samples
                    if (coordinates.Length > 0) // Avoid division by zero
                    {
                        for (int i = 0; i < coefficients.Length; i++)
                        {
                            coefficients[i] /= coordinates.Length;
                        }
                    }
                    return coefficients;
                }

                /// <summary>
                /// Calculates the eigenvalues of the curvature tensor.
                /// </summary>
                float[] CalculateEigenvalues(float[] coefficients)
                {
                    if (coefficients == null || coefficients.Length < 6)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: Invalid input for CalculateEigenvalues. Returning default eigenvalues.");
                        return new float[] { 1.0f, 1.0f, 1.0f }; // Return default values
                    }


                    // Construct a simplified 3x3 matrix from the first 6 coefficients
                    float[,] matrix = new float[3, 3];
                    matrix[0, 0] = coefficients[0]; // xx
                    matrix[1, 1] = coefficients[1]; // yy
                    matrix[2, 2] = coefficients[2]; // zz
                    matrix[0, 1] = matrix[1, 0] = coefficients[3]; // xy
                    matrix[0, 2] = matrix[2, 0] = coefficients[4]; // xz
                    matrix[1, 2] = matrix[2, 1] = coefficients[5]; // yz

                    // Compute the trace (sum of diagonal elements)
                    float trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2];

                    // Compute the determinant
                    float det = matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1]) -
                                        matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0]) +
                                        matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0]);

                    // Use a simplified approach to estimate eigenvalues
                    float[] eigenvalues = new float[3];
                    // Slightly different calculation than Unit A
                    eigenvalues[0] = trace / 3.0f + 0.12f * det; // Approximation for first eigenvalue
                    eigenvalues[1] = trace / 3.0f - 0.03f * det; // Approximation for second eigenvalue
                    eigenvalues[2] = trace / 3.0f - 0.09f * det; // Approximation for third eigenvalue


                    return eigenvalues;
                }

                /// <summary>
                /// Converts eigenvalues to weights for loss function.
                /// </summary>
                float[] ConvertEigenvaluesToWeights(float[] eigenvalues)
                {
                    if (eigenvalues == null || eigenvalues.Length == 0)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: Eigenvalues array is null or empty for weights. Returning default weights.");
                        return new float[] { 1.0f }; // Return a default weight
                    }

                    // Create weights based on eigenvalues
                    float[] weights = new float[eigenvalues.Length];

                    // Normalize eigenvalues to create weights that sum to something meaningful
                    // Use absolute values, and shift/scale to ensure positive weights
                    float sumAbsEigenvalues = 0.0f;
                    for (int i = 0; i < eigenvalues.Length; i++)
                    {
                        // Use absolute values to ensure positive weights
                        weights[i] = Math.Abs(eigenvalues[i]);
                        sumAbsEigenvalues += weights[i];
                    }

                    // Normalize and ensure minimum weight, or use a relative weighting
                    if (sumAbsEigenvalues > 1e-6f) // Use tolerance
                    {
                        // Use a relative weighting: higher absolute eigenvalue means higher weight
                        float maxAbsEigenvalue = weights.Max();
                        if (maxAbsEigenvalue > 1e-6f)
                        {
                            for (int i = 0; i < weights.Length; i++)
                            {
                                // Scale weights relative to max, with a minimum base value
                                weights[i] = 0.5f + 0.5f * (weights[i] / maxAbsEigenvalue); // Weights between 0.5 and 1.0
                            }
                        }
                        else
                        {
                            // Fallback if max is zero (all eigenvalues are zero)
                            for (int i = 0; i < weights.Length; i++)
                            {
                                weights[i] = 1.0f; // Equal weights
                            }
                        }

                        // If we only need a single scalar weight for the overall loss
                        // For the loss function structure `tf.reduce_mean(tf.multiply(rawLoss, curvatureWeightTensor))`
                        // where rawLoss is (batch_size, 1), curvatureWeightTensor should also be (batch_size, 1) or (1, 1).
                        // Let's average the calculated weights to get a single scalar weight for the batch loss.
                        return new float[] { weights.Average() }; // Return average weight as a scalar array
                    }
                    else
                    {
                        // Default equal weights if eigenvalues sum to zero (all eigenvalues are zero or near zero)
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: Sum of absolute eigenvalues is zero. Returning default weights.");
                        return new float[] { 1.0f }; // Return a default scalar weight
                    }
                }


                /// <summary>
                /// Calculates a mask that identifies the outermost vertices in a tensor.
                /// This version is a simplified conceptual mask for a 2D tensor.
                /// </summary>
                Tensor CalculateOutermostVertexMask(Tensor input)
                {
                    if (input == null || input.shape.rank < 2)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper (Unit B): CalculateOutermostVertexMask received null or insufficient rank tensor. Returning default mask.");
                        return tf.ones_like(input); // Return identity mask
                    }
                    // Get the shape of the input tensor
                    var shape = tf.shape(input); // Shape is (batch_size, features)
                    var batchSize = tf.slice(shape, begin: new int[] { 0 }, size: new int[] { 1 });
                    var features = tf.slice(shape, begin: new int[] { 1 }, size: new int[] { 1 });

                    // Create a feature-wise weight that emphasizes the ends (conceptual vertices)
                    var featureIndices = tf.cast(tf.range(0, features), dtype: tf.float32); // Indices 0 to features-1
                    var normalizedIndices = tf.divide(featureIndices, tf.cast(features - 1, tf.float32)); // Normalize to 0-1

                    // Use a pattern that is high at 0 and 1, low in the middle
                    // abs(normalizedIndices - 0.5) gives values from 0.5 to 0 then back to 0.5
                    // Multiplying by 2 gives values from 1.0 to 0 then back to 1.0
                    var featureMask = tf.multiply(tf.abs(normalizedIndices - 0.5f), 2.0f, name: "feature_vertex_mask_B"); // Shape (features,)


                    // Expand the mask to match the batch dimension (batch_size, features)
                    var batchSizeInt = tf.cast(batchSize, tf.int32);
                    // Tile requires an array for multiples
                    var expandedMask = tf.tile(tf.reshape(featureMask, shape: new int[] { 1, -1 }), multiples: tf.concat(new[] { batchSizeInt, tf.constant(new int[] { 1 }) }, axis: 0), name: "expanded_vertex_mask_B");

                    return expandedMask;
                }


                /// <summary>
                /// Converts a jagged array to a multidimensional array.
                /// </summary>
                float[,] ConvertJaggedToMultidimensional(float[][] jaggedArray)
                {
                    if (jaggedArray == null || jaggedArray.Length == 0 || jaggedArray.Any(row => row == null))
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper (Unit B): ConvertJaggedToMultidimensional received null, empty, or jagged array with null rows. Returning empty multidimensional array.");
                        return new float[0, 0];
                    }
                    if (jaggedArray[0].Length == 0)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper (Unit B): ConvertJaggedToMultidimensional received jagged array with zero columns. Returning empty multidimensional array.");
                        return new float[jaggedArray.Length, 0];
                    }

                    int rows = jaggedArray.Length;
                    int cols = jaggedArray[0].Length;

                    float[,] result = new float[rows, cols];

                    for (int i = 0; i < rows; i++)
                    {
                        if (jaggedArray[i].Length != cols)
                        {
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning (Unit B): Row {i} in jagged array has inconsistent length ({jaggedArray[i].Length} vs {cols}). Returning partial result.");
                            int currentCols = jaggedArray[i].Length;
                            for (int j = 0; j < Math.Min(cols, currentCols); j++)
                            {
                                result[i, j] = jaggedArray[i][j];
                            }
                        }
                        else
                        {
                            System.Buffer.BlockCopy(jaggedArray[i], 0, result, i * cols * sizeof(float), cols * sizeof(float)); // Use System.Buffer
                        }
                    }

                    return result;
                }


                /// <summary>
                /// Extracts a batch from a multidimensional array using indices.
                /// </summary>
                float[,] ExtractBatch(float[,] data, int[] batchIndices, int startIdx, int count)
                {
                    if (data == null || batchIndices == null || data.GetLength(0) == 0 || batchIndices.Length == 0)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning (Unit B): Invalid or empty input data/indices for ExtractBatch. Returning empty batch.");
                        return new float[0, data?.GetLength(1) ?? 0];
                    }
                    if (batchIndices.Length < startIdx + count || (batchIndices.Length > 0 && data.GetLength(0) <= batchIndices.Max()))
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning (Unit B): Indices or data size mismatch for ExtractBatch (data rows: {data.GetLength(0)}, indices length: {batchIndices.Length}, startIdx: {startIdx}, count: {count}, max index in batch: {(batchIndices.Length > 0 ? batchIndices.Skip(startIdx).Take(count).DefaultIfEmpty(-1).Max() : -1)}). Returning empty batch.");
                        return new float[0, data.GetLength(1)];
                    }


                    if (count <= 0) return new float[0, data.GetLength(1)]; // Return empty batch if count is non-positive

                    int cols = data.GetLength(1);
                    float[,] batch = new float[count, cols];

                    for (int i = 0; i < count; i++)
                    {
                        int srcIdx = batchIndices[startIdx + i]; // Used batchIndices
                        if (srcIdx < 0 || srcIdx >= data.GetLength(0))
                        {
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Error (Unit B): Invalid index {srcIdx} in ExtractBatch indices array at batch index {i}. Stopping batch extraction.");
                            var partialBatch = new float[i, cols];
                            System.Buffer.BlockCopy(batch, 0, partialBatch, 0, i * cols * sizeof(float)); // Use System.Buffer
                            return partialBatch;
                        }
                        System.Buffer.BlockCopy(data, srcIdx * cols * sizeof(float), batch, i * cols * sizeof(float), cols * sizeof(float)); // Use System.Buffer
                    }

                    return batch;
                }

                /// <summary>
                /// Shuffles an array randomly.
                /// </summary>
                void ShuffleArray(int[] shuffleIndices)
                {
                    if (shuffleIndices == null) return;
                    Random rng = new Random();
                    int n = shuffleIndices.Length;

                    while (n > 1)
                    {
                        n--;
                        int k = rng.Next(n + 1);
                        int temp = shuffleIndices[k];
                        shuffleIndices[k] = shuffleIndices[n];
                        shuffleIndices[n] = temp;
                    }
                }

                /// <summary>
                /// Serializes model metadata to JSON.
                /// </summary>
                string SerializeMetadata(Dictionary<string, object> metadata)
                {
                    if (metadata == null) return "{}";
                    StringBuilder sb = new StringBuilder();
                    sb.Append("{");
                    bool first = true;
                    foreach (var entry in metadata)
                    {
                        if (!first) sb.Append(",");
                        sb.Append($"\"{entry.Key}\":");
                        if (entry.Value is string strValue) sb.Append($"\"{strValue.Replace("\"", "\\\"")}\"");
                        else if (entry.Value is float[] floatArray) sb.Append($"[{string.Join(",", floatArray.Select(f => f.ToString("F6")))}]");
                        else if (entry.Value is double[] doubleArray) sb.Append($"[{string.Join(",", doubleArray.Select(d => d.ToString("F6")))}]");
                        else if (entry.Value is int[] intArray) sb.Append($"[{string.Join(",", intArray)}]");
                        else if (entry.Value is System.Numerics.Vector3 vector) sb.Append($"[\"{vector.X:F6}\",\"{vector.Y:F6}\",\"{vector.Z:F6}\"]");
                        else if (entry.Value is float floatValue) sb.Append(floatValue.ToString("F6"));
                        else if (entry.Value is double doubleValue) sb.Append(doubleValue.ToString("F6"));
                        else if (entry.Value is int intValue) sb.Append(intValue.ToString());
                        else if (entry.Value is bool boolValue) sb.Append(boolValue.ToString().ToLower());
                        else if (entry.Value is float[,] float2DArray)
                        {
                            sb.Append("[");
                            for (int i = 0; i < float2DArray.GetLength(0); i++)
                            {
                                if (i > 0) sb.Append(",");
                                sb.Append("[");
                                for (int j = 0; j < float2DArray.GetLength(1); j++)
                                {
                                    if (j > 0) sb.Append(",");
                                    sb.Append(float2DArray[i, j].ToString("F6"));
                                }
                                sb.Append("]");
                            }
                            sb.Append("]");
                        }
                        else if (entry.Value != null)
                        {
                            try { string valueStr = entry.Value.ToString(); if (float.TryParse(valueStr, out _) || double.TryParse(valueStr, out _) || bool.TryParse(valueStr, out _)) sb.Append(valueStr); else sb.Append($"\"{valueStr.Replace("\"", "\\\"")}\""); }
                            catch (Exception ex) { Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning: Failed to serialize metadata value for key '{entry.Key}': {ex.Message}"); sb.Append("\"SerializationError\""); }
                        }
                        else sb.Append("null");
                        first = false;
                    }
                    sb.Append("}");
                    return sb.ToString();
                }


                // Helper function to process an array with K-means clustering
                void ProcessArrayWithKMeans(double[] dataArray, string arrayName, ConcurrentDictionary<string, object> resultsStore)
                {
                    string unitBArrayName = arrayName + "_B"; // Append _B for Unit B's results
                    if (dataArray == null || dataArray.Length < 3 || dataArray.All(d => d == dataArray[0]))
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Not enough distinct data points in {unitBArrayName} for K-means clustering or data is constant. Skipping.");
                        resultsStore[$"{unitBArrayName}_Category"] = "InsufficientData";
                        resultsStore[$"{unitBArrayName}_NormalizedValue"] = 0.0;
                        resultsStore[$"{unitBArrayName}_NormalizedX"] = 0.0;
                        resultsStore[$"{unitBArrayName}_NormalizedY"] = 0.0;
                        resultsStore[$"{unitBArrayName}_NormalizedZ"] = 0.0;
                        return;
                    }
                    try
                    {
                        double[][] points = new double[dataArray.Length][];
                        for (int i = 0; i < dataArray.Length; i++) points[i] = new double[] { dataArray[i] };
                        int k = Math.Min(3, points.Length); if (k < 1) k = 1;
                        var kmeans = new Accord.MachineLearning.KMeans(k) { Distance = new Accord.Math.Distances.SquareEuclidean() };
                        try
                        {
                            var clusters = kmeans.Learn(points);
                            double[] centroids = clusters.Centroids.Select(c => c[0]).ToArray();
                            Array.Sort(centroids); Array.Reverse(centroids);
                            int numCentroids = Math.Min(3, centroids.Length);
                            double[] paddedCentroids = new double[3];
                            for (int i = 0; i < numCentroids; i++) paddedCentroids[i] = centroids[i];
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] K-means centroids for {unitBArrayName}: [{string.Join(", ", centroids.Take(numCentroids).Select(c => c.ToString("F4")))}]");
                            double centralPoint = centroids.Take(numCentroids).DefaultIfEmpty(0).Average();
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Central point for {unitBArrayName}: {centralPoint}");
                            double maxAbsCentroid = centroids.Take(numCentroids).Select(Math.Abs).DefaultIfEmpty(0).Max();
                            double normalizedValue = maxAbsCentroid > 1e-6 ? centralPoint / maxAbsCentroid : 0;
                            string category;
                            if (normalizedValue < -0.33) category = "Negative High";
                            else if (normalizedValue < 0) category = "Negative Low";
                            else if (normalizedValue < 0.33) category = "Positive Low";
                            else if (normalizedValue < 0.66) category = "Positive Medium";
                            else category = "Positive High";
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Normalized value for {unitBArrayName}: {normalizedValue:F4}, Category: {category}");
                            double x = paddedCentroids[0], y = paddedCentroids[1], z = paddedCentroids[2];
                            double maxAbsCoordinate = Math.Max(Math.Max(Math.Abs(x), Math.Abs(y)), Math.Abs(z));
                            if (maxAbsCoordinate > 1e-6) { x /= maxAbsCoordinate; y /= maxAbsCoordinate; z /= maxAbsCoordinate; }
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Normalized XYZ coordinates for {unitBArrayName}: ({x:F4}, {y:F4}, {z:F4})");
                            resultsStore[$"{unitBArrayName}_Category"] = category;
                            resultsStore[$"{unitBArrayName}_NormalizedValue"] = normalizedValue;
                            resultsStore[$"{unitBArrayName}_NormalizedX"] = x;
                            resultsStore[$"{unitBArrayName}_NormalizedY"] = y;
                            resultsStore[$"{unitBArrayName}_NormalizedZ"] = z;
                        }
                        catch (Exception learnEx)
                        {
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Error during K-means Learn for {unitBArrayName}: {learnEx.Message}");
                            resultsStore[$"{unitBArrayName}_Category"] = "ClusteringLearnError"; resultsStore[$"{unitBArrayName}_NormalizedValue"] = 0.0; resultsStore[$"{unitBArrayName}_NormalizedX"] = 0.0; resultsStore[$"{unitBArrayName}_NormalizedY"] = 0.0; resultsStore[$"{unitBArrayName}_NormalizedZ"] = 0.0;
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Error processing K-means for {unitBArrayName}: {ex.Message}");
                        resultsStore[$"{unitBArrayName}_Category"] = "ProcessingError"; resultsStore[$"{unitBArrayName}_NormalizedValue"] = 0.0; resultsStore[$"{unitBArrayName}_NormalizedX"] = 0.0; resultsStore[$"{unitBArrayName}_NormalizedY"] = 0.0; resultsStore[$"{unitBArrayName}_NormalizedZ"] = 0.0;
                    }
                }

                // Helper function to calculate trajectory stability
                double CalculateTrajectoryStability(List<double[]> trajectoryPoints)
                {
                    if (trajectoryPoints == null || trajectoryPoints.Count < 2) return 0.5;
                    double averageAngleChange = 0; int angleCount = 0;
                    for (int i = 1; i < trajectoryPoints.Count - 1; i++)
                    {
                        if (trajectoryPoints[i - 1] == null || trajectoryPoints[i - 1].Length < 3 || trajectoryPoints[i] == null || trajectoryPoints[i].Length < 3 || trajectoryPoints[i + 1] == null || trajectoryPoints[i + 1].Length < 3) continue;
                        double[] prevVector = new double[3], nextVector = new double[3];
                        for (int j = 0; j < 3; j++) prevVector[j] = trajectoryPoints[i][j] - trajectoryPoints[i - 1][j];
                        for (int j = 0; j < 3; j++) nextVector[j] = trajectoryPoints[i + 1][j] - trajectoryPoints[i][j];
                        double dotProduct = 0, prevMagSq = 0, nextMagSq = 0;
                        for (int j = 0; j < 3; j++) { dotProduct += prevVector[j] * nextVector[j]; prevMagSq += prevVector[j] * prevVector[j]; nextMagSq += nextVector[j] * nextVector[j]; }
                        double prevMag = Math.Sqrt(prevMagSq), nextMag = Math.Sqrt(nextMagSq);
                        if (prevMag > 1e-9 && nextMag > 1e-9)
                        {
                            double cosAngle = dotProduct / (prevMag * nextMag);
                            cosAngle = Math.Max(-1.0, Math.Min(1.0, cosAngle));
                            double angle = Math.Acos(cosAngle);
                            averageAngleChange += angle; angleCount++;
                        }
                    }
                    if (angleCount > 0) averageAngleChange /= angleCount; else return 0.5;
                    return 1.0 - (averageAngleChange / Math.PI);
                }

                // Helper function to calculate plane intersection point
                double[] CalculatePlaneIntersection(List<double[]> trajectoryPoints, int planeAxis, double tolerance)
                {
                    if (trajectoryPoints == null || trajectoryPoints.Count < 2) return null;
                    for (int i = 1; i < trajectoryPoints.Count; i++)
                    {
                        if (trajectoryPoints[i - 1] == null || trajectoryPoints[i - 1].Length < Math.Max(3, planeAxis + 1) || trajectoryPoints[i] == null || trajectoryPoints[i].Length < Math.Max(3, planeAxis + 1)) continue;
                        double v1 = trajectoryPoints[i - 1][planeAxis], v2 = trajectoryPoints[i][planeAxis];
                        bool crossedZero = (v1 >= -tolerance && v2 <= tolerance) || (v1 <= tolerance && v2 >= -tolerance);
                        if (crossedZero && !((v1 >= -tolerance && v1 <= tolerance) && (v2 >= -tolerance && v2 <= tolerance)))
                        {
                            if (Math.Abs(v1) < tolerance) return (double[])trajectoryPoints[i - 1].Clone();
                            if (Math.Abs(v2) < tolerance) return (double[])trajectoryPoints[i].Clone();
                            double t = Math.Abs(v1) / (Math.Abs(v1) + Math.Abs(v2));
                            double[] intersection = new double[3];
                            for (int j = 0; j < 3; j++) { if (j < trajectoryPoints[i - 1].Length && j < trajectoryPoints[i].Length) intersection[j] = trajectoryPoints[i - 1][j] * (1 - t) + trajectoryPoints[i][j] * t; else intersection[j] = 0; }
                            if (planeAxis >= 0 && planeAxis < intersection.Length) intersection[planeAxis] = 0.0;
                            return intersection;
                        }
                    }
                    return null;
                }

                // Helper functions to count negative points using tolerance
                int CountNegativePoints(List<double[]> points, int axis, double tolerance)
                {
                    if (points == null) return 0; int count = 0;
                    foreach (var point in points) { if (point != null && point.Length > axis && point[axis] < -tolerance) count++; }
                    return count;
                }

                int CountNegativeBothPoints(List<double[]> points, double tolerance)
                {
                    if (points == null) return 0; int count = 0;
                    foreach (var point in points) { if (point != null && point.Length >= 2 && point[0] < -tolerance && point[1] < -tolerance) count++; }
                    return count;
                }

                double[] FindPositiveCoordinate(List<double[]> points, double tolerance)
                {
                    if (points == null || points.Count == 0) return new double[] { 0, 0, 0 };
                    foreach (var point in points) { if (point != null && point.Length >= 2 && point[0] > tolerance && point[1] > tolerance) return (double[])point.Clone(); }
                    int firstIndex = points.Count > 0 ? 0 : -1;
                    if (firstIndex != -1 && points[firstIndex] != null && points[firstIndex].Length >= 3) return (double[])points[firstIndex].Clone();
                    return new double[] { 0, 0, 0 };
                }

                double[] FindNegativeCoordinate(List<double[]> points, double tolerance)
                {
                    if (points == null || points.Count == 0) return new double[] { 0, 0, 0 };
                    foreach (var point in points) { if (point != null && point.Length >= 2 && point[0] < -tolerance && point[1] < -tolerance) return (double[])point.Clone(); }
                    int lastIndex = points.Count > 0 ? points.Count - 1 : 0;
                    if (lastIndex >= 0 && points[lastIndex] != null && points[lastIndex].Length >= 3) return (double[])points[lastIndex].Clone();
                    return new double[] { 0, 0, 0 };
                }

                double CalculateVelocity(double[] trajectory, double magnitude) { return magnitude; }

                #endregion // End of Helper Methods for Unit B


                //==========================================================================
                // Workflow Coordination
                //==========================================================================
                // This function coordinates the sequential execution of the eight processing stages.
                string ExecuteProductionWorkflow_B(CoreMlOutcomeRecord record, int custId, Tensorflow.Session session_param_unused) // session param is now unused here
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Starting multi-stage workflow (Unit B) for customer {custId}.");

                    // Step 1: Begin Data Acquisition and Initial Feature Analysis
                    string analysisResult = Stage1_DataAcquisitionAndAnalysis_B(record, custId);
                    unitResultsStore["B_DataAcquisitionResult"] = analysisResult;

                    // Step 2: Begin Feature Tensor Generation and Trajectory Mapping
                    string tensorMappingResult = Stage2_FeatureTensorAndMapping_B(analysisResult, custId);
                    unitResultsStore["B_FeatureTensorMappingResult"] = tensorMappingResult;

                    // Step 3: Begin Processed Feature Definition Creation
                    string processedFeatureResult = Stage3_ProcessedFeatureDefinition_B(tensorMappingResult, custId); // Removed session from params
                    unitResultsStore["B_ProcessedFeatureResult"] = processedFeatureResult;

                    // Step 4: Begin Feature Quality Assessment
                    string qualityAssessmentResult = Stage4_FeatureQualityAssessment_B(processedFeatureResult, custId);
                    unitResultsStore["B_QualityAssessmentResult"] = qualityAssessmentResult;

                    // Step 5: Begin Combined Feature Evaluation
                    float combinedEvaluationScore = Stage5_CombinedFeatureEvaluation_B(qualityAssessmentResult, custId);
                    unitResultsStore["B_CombinedEvaluationScore"] = combinedEvaluationScore;

                    // Step 6: Begin Fractal Optimization Analysis
                    string optimizationAnalysisResult = Stage6_FractalOptimizationAnalysis_B(qualityAssessmentResult, combinedEvaluationScore, custId);
                    unitResultsStore["B_OptimizationAnalysisResult"] = optimizationAnalysisResult;

                    // Step 7: Begin Tensor Network Training with Curvature Embedding (Includes Actual TF.NET)
                    // Model B now creates its own session internally.
                    // The passed 'session_param_unused' (originally mlSession) is not used for TF ops in Stage7.
                    string trainingOutcomeResult = Stage7_TensorNetworkTraining_B(optimizationAnalysisResult, custId, unitResultsStore);
                    unitResultsStore["B_TensorNetworkTrainingOutcome"] = trainingOutcomeResult;

                    // Step 8: Begin Future Performance Projection
                    string performanceProjectionResult = Stage8_FutureProjection_B(trainingOutcomeResult, combinedEvaluationScore, custId);
                    unitResultsStore["B_PerformanceProjectionResult"] = performanceProjectionResult;


                    // Final workflow result combines key outputs - Use the projected score as the final outcome
                    float finalScore = unitResultsStore.TryGetValue("B_ProjectedPerformanceScore", out var projectedScoreVal) && projectedScoreVal is float projectedFloat ? projectedFloat : combinedEvaluationScore;


                    // Make sure to store the score with BOTH keys - the unit-specific and the standard key expected by Unit D
                    unitResultsStore["ModelBProcessingOutcome"] = finalScore; // Standard key for Unit D
                    // unitResultsStore["B_FinalScore"] = finalScore; // Unit-specific key if needed

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Workflow (Unit B) completed for customer {custId} with final score {finalScore:F4}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: ModelBProcessingOutcome stored: {unitResultsStore.ContainsKey("ModelBProcessingOutcome")}");

                    return $"Workflow_B_Complete_Cust_{custId}_FinalScore_{finalScore:F4}";
                }

                //==========================================================================
                // Step 1 (Unit B): Data Acquisition & Analysis
                //==========================================================================
                string Stage1_DataAcquisitionAndAnalysis_B(CoreMlOutcomeRecord record, int custId)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 1 (Unit B) - Acquiring data and analyzing initial features for customer {custId}.");

                    var productInventory = RuntimeProcessingContext.RetrieveContextValue("All_Simulated_Product_Inventory") as List<dynamic>;
                    var serviceOfferings = RuntimeProcessingContext.RetrieveContextValue("All_Simulated_Service_Offerings") as List<dynamic>;

                    if (productInventory != null)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Step 1 (Unit B) - Processing Product Data ({productInventory.Count} items).");
                        var quantityAvailable = new List<int>();
                        var productMonetaryValue = new List<double>();
                        var productCostContribution = new List<double>();

                        foreach (var product in productInventory)
                        {
                            try
                            {
                                quantityAvailable.Add(Convert.ToInt32(product.QuantityAvailable));
                                productMonetaryValue.Add(Convert.ToDouble(product.MonetaryValue));
                                productCostContribution.Add(Convert.ToDouble(product.CostContributionValue));
                            }
                            catch (Microsoft.CSharp.RuntimeBinder.RuntimeBinderException rbEx)
                            {
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unit B RuntimeBinder Error accessing product properties: {rbEx.Message}");
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unit B Unexpected Error accessing product properties: {ex.Message}");
                            }
                        }

                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unit B Product QuantityAvailable: [{string.Join(", ", quantityAvailable)}]");
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unit B Product MonetaryValue: [{string.Join(", ", productMonetaryValue)}]");
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unit B Product CostContributionValue: [{string.Join(", ", productCostContribution)}]");

                        ProcessArrayWithKMeans(quantityAvailable.Select(x => (double)x).ToArray(), "Product QuantityAvailable", unitResultsStore);
                        ProcessArrayWithKMeans(productMonetaryValue.ToArray(), "Product MonetaryValue", unitResultsStore);
                        ProcessArrayWithKMeans(productCostContribution.ToArray(), "Product CostContributionValue", unitResultsStore);
                    }
                    else
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unit B Product inventory data not found in RuntimeProcessingContext. Skipping product analysis.");
                    }

                    if (serviceOfferings != null)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Step 1 (Unit B) - Processing Service Data ({serviceOfferings.Count} items).");
                        var fulfillmentQuantity = new List<int>();
                        var serviceMonetaryValue = new List<double>();
                        var serviceCostContribution = new List<double>();

                        foreach (var service in serviceOfferings)
                        {
                            try
                            {
                                fulfillmentQuantity.Add(Convert.ToInt32(service.FulfillmentQuantity));
                                serviceMonetaryValue.Add(Convert.ToDouble(service.MonetaryValue));
                                serviceCostContribution.Add(Convert.ToDouble(service.CostContributionValue));
                            }
                            catch (Microsoft.CSharp.RuntimeBinder.RuntimeBinderException rbEx)
                            {
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unit B RuntimeBinder Error accessing service properties: {rbEx.Message}");
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unit B Unexpected Error accessing service properties: {ex.Message}");
                            }
                        }

                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unit B Service FulfillmentQuantity: [{string.Join(", ", fulfillmentQuantity)}]");
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unit B Service MonetaryValue: [{string.Join(", ", serviceMonetaryValue)}]");
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unit B Service CostContributionValue: [{string.Join(", ", serviceCostContribution)}]");

                        ProcessArrayWithKMeans(fulfillmentQuantity.Select(x => (double)x).ToArray(), "Service FulfillmentQuantity", unitResultsStore);
                        ProcessArrayWithKMeans(serviceMonetaryValue.ToArray(), "Service MonetaryValue", unitResultsStore);
                        ProcessArrayWithKMeans(serviceCostContribution.ToArray(), "Service CostContributionValue", unitResultsStore);
                    }
                    else
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unit B Service offerings data not found in RuntimeProcessingContext. Skipping service analysis.");
                    }

                    string result = $"InitialAnalysis_B_Cust_{custId}_Record_{record.RecordIdentifier}";
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 1 (Unit B) - Data acquisition and initial analysis completed: {result}");
                    return result;
                }

                //==========================================================================
                // Step 2 (Unit B): Feature Tensor Generation & Mapping
                //==========================================================================
                string Stage2_FeatureTensorAndMapping_B(string analysisResult, int custId)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 2 (Unit B) - Generating feature tensors and mapping trajectories for customer {custId}.");

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Step 2 (Unit B) - Retrieving coordinates from Step 1 analysis.");

                    string prodQtyCategory = unitResultsStore.TryGetValue("Product QuantityAvailable_B_Category", out var pqc) ? pqc.ToString() : "Unknown";
                    double prodQtyX = unitResultsStore.TryGetValue("Product QuantityAvailable_B_NormalizedX", out var pqx) ? Convert.ToDouble(pqx) : 0;
                    double prodQtyY = unitResultsStore.TryGetValue("Product QuantityAvailable_B_NormalizedY", out var pqy) ? Convert.ToDouble(pqy) : 0;
                    double prodQtyZ = unitResultsStore.TryGetValue("Product QuantityAvailable_B_NormalizedZ", out var pqz) ? Convert.ToDouble(pqz) : 0;

                    string prodMonCategory = unitResultsStore.TryGetValue("Product MonetaryValue_B_Category", out var pmc) ? pmc.ToString() : "Unknown";
                    double prodMonX = unitResultsStore.TryGetValue("Product MonetaryValue_B_NormalizedX", out var pmx) ? Convert.ToDouble(pmx) : 0;
                    double prodMonY = unitResultsStore.TryGetValue("Product MonetaryValue_B_NormalizedY", out var pmy) ? Convert.ToDouble(pmy) : 0;
                    double prodMonZ = unitResultsStore.TryGetValue("Product MonetaryValue_B_NormalizedZ", out var pmz) ? Convert.ToDouble(pmz) : 0;

                    string prodCostCategory = unitResultsStore.TryGetValue("Product CostContributionValue_B_Category", out var pcc) ? pcc.ToString() : "Unknown";
                    double prodCostX = unitResultsStore.TryGetValue("Product CostContributionValue_B_NormalizedX", out var pcx) ? Convert.ToDouble(pcx) : 0;
                    double prodCostY = unitResultsStore.TryGetValue("Product CostContributionValue_B_NormalizedY", out var pcy) ? Convert.ToDouble(pcy) : 0;
                    double prodCostZ = unitResultsStore.TryGetValue("Product CostContributionValue_B_NormalizedZ", out var pcz) ? Convert.ToDouble(pcz) : 0;

                    string servFulfillCategory = unitResultsStore.TryGetValue("Service FulfillmentQuantity_B_Category", out var sfc) ? sfc.ToString() : "Unknown";
                    double servFulfillX = unitResultsStore.TryGetValue("Service FulfillmentQuantity_B_NormalizedX", out var sfx) ? Convert.ToDouble(sfx) : 0;
                    double servFulfillY = unitResultsStore.TryGetValue("Service FulfillmentQuantity_B_NormalizedY", out var sfy) ? Convert.ToDouble(sfy) : 0;
                    double servFulfillZ = unitResultsStore.TryGetValue("Service FulfillmentQuantity_B_NormalizedZ", out var sfz) ? Convert.ToDouble(sfz) : 0;

                    string servMonCategory = unitResultsStore.TryGetValue("Service MonetaryValue_B_Category", out var smc) ? smc.ToString() : "Unknown";
                    double servMonX = unitResultsStore.TryGetValue("Service MonetaryValue_B_NormalizedX", out var smx) ? Convert.ToDouble(smx) : 0;
                    double servMonY = unitResultsStore.TryGetValue("Service MonetaryValue_B_NormalizedY", out var smy) ? Convert.ToDouble(smy) : 0;
                    double servMonZ = unitResultsStore.TryGetValue("Service MonetaryValue_B_NormalizedZ", out var smz) ? Convert.ToDouble(smz) : 0;

                    string servCostCategory = unitResultsStore.TryGetValue("Service CostContributionValue_B_Category", out var scc) ? scc.ToString() : "Unknown";
                    double servCostX = unitResultsStore.TryGetValue("Service CostContributionValue_B_NormalizedX", out var scx) ? Convert.ToDouble(scx) : 0;
                    double servCostY = unitResultsStore.TryGetValue("Service CostContributionValue_B_NormalizedY", out var scy) ? Convert.ToDouble(scy) : 0;
                    double servCostZ = unitResultsStore.TryGetValue("Service CostContributionValue_B_NormalizedZ", out var scz) ? Convert.ToDouble(scz) : 0;

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Step 2 (Unit B) - Calculating tensors, magnitudes, and trajectories.");

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] ----- PRODUCT TENSOR AND MAGNITUDE CALCULATIONS (Unit B) -----");
                    double prodOverallTensorX = (prodQtyX + prodMonX + prodCostX) / 3.0;
                    double prodOverallTensorY = (prodQtyY + prodMonY + prodCostY) / 3.0;
                    double prodOverallTensorZ = (prodQtyZ + prodMonZ + prodCostZ) / 3.0;
                    double prodOverallMagnitude = Math.Sqrt(prodOverallTensorX * prodOverallTensorX + prodOverallTensorY * prodOverallTensorY + prodOverallTensorZ * prodOverallTensorZ);
                    double[] prodTrajectory = new double[3] { 0, 0, 0 };
                    if (prodOverallMagnitude > 1e-9)
                    {
                        prodTrajectory[0] = prodOverallTensorX / prodOverallMagnitude;
                        prodTrajectory[1] = prodOverallTensorY / prodOverallMagnitude;
                        prodTrajectory[2] = prodOverallTensorZ / prodOverallMagnitude;
                    }
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product Overall Tensor (Unit B): ({prodOverallTensorX:F4}, {prodOverallTensorY:F4}, {prodOverallTensorZ:F4})");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product Overall Magnitude (Unit B): {prodOverallMagnitude:F4}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product Trajectory (Unit B): ({prodTrajectory[0]:F4}, {prodTrajectory[1]:F4}, {prodTrajectory[2]:F4})");

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] ----- SERVICE TENSOR AND MAGNITUDE CALCULATIONS (Unit B) -----");
                    double servOverallTensorX = (servFulfillX + servMonX + servCostX) / 3.0;
                    double servOverallTensorY = (servFulfillY + servMonY + servCostY) / 3.0;
                    double servOverallTensorZ = (servFulfillZ + servMonZ + servCostZ) / 3.0;
                    double servOverallMagnitude = Math.Sqrt(servOverallTensorX * servOverallTensorX + servOverallTensorY * servOverallTensorY + servOverallTensorZ * servOverallTensorZ);
                    double[] servTrajectory = new double[3] { 0, 0, 0 };
                    if (servOverallMagnitude > 1e-9)
                    {
                        servTrajectory[0] = servOverallTensorX / servOverallMagnitude;
                        servTrajectory[1] = servOverallTensorY / servOverallMagnitude;
                        servTrajectory[2] = servOverallTensorZ / servOverallMagnitude;
                    }
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service Overall Tensor (Unit B): ({servOverallTensorX:F4}, {servOverallTensorY:F4}, {servOverallTensorZ:F4})");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service Overall Magnitude (Unit B): {servOverallMagnitude:F4}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service Trajectory (Unit B): ({servTrajectory[0]:F4}, {servTrajectory[1]:F4}, {servTrajectory[2]:F4})");

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] ----- TRAJECTORY PLOT GENERATION & ANALYSIS (Unit B) -----");
                    const int MAX_RECURSION_DEPTH = 40; // Different from Unit A for variation
                    const double CONTINUE_PAST_PLANE = -1.5; // Different from Unit A
                    List<double[]> productTrajectoryPoints = new List<double[]>();
                    List<double> productPointIntensities = new List<double>();
                    List<double[]> serviceTrajectoryPoints = new List<double[]>();
                    List<double> servicePointIntensities = new List<double>();
                    double[] productCurrentPosition = new double[] { prodOverallTensorX, prodOverallTensorY, prodOverallTensorZ };
                    double[] serviceCurrentPosition = new double[] { servOverallTensorX, servOverallTensorY, servOverallTensorZ };
                    double recursionFactor = 0.90; // Different from Unit A

                    double[] InvertTrajectoryIfNeeded(double[] trajectory)
                    {
                        bool movesTowardNegativeX = trajectory != null && trajectory.Length > 0 && trajectory[0] < -1e-6;
                        bool movesTowardNegativeY = trajectory != null && trajectory.Length > 1 && trajectory[1] < -1e-6;
                        if (!movesTowardNegativeX || !movesTowardNegativeY)
                        {
                            if (trajectory == null || trajectory.Length < 3) return new double[] { 0, 0, 0 };
                            double[] invertedTrajectory = new double[3];
                            invertedTrajectory[0] = movesTowardNegativeX ? trajectory[0] : -Math.Abs(trajectory[0]);
                            invertedTrajectory[1] = movesTowardNegativeY ? trajectory[1] : -Math.Abs(trajectory[1]);
                            invertedTrajectory[2] = trajectory.Length > 2 ? trajectory[2] : 0;
                            double magnitude = Math.Sqrt(invertedTrajectory[0] * invertedTrajectory[0] + invertedTrajectory[1] * invertedTrajectory[1] + invertedTrajectory[2] * invertedTrajectory[2]);
                            if (magnitude > 1e-9)
                            {
                                invertedTrajectory[0] /= magnitude;
                                invertedTrajectory[1] /= magnitude;
                                invertedTrajectory[2] /= magnitude;
                            }
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unit B Inverted trajectory from ({trajectory[0]:F4}, {trajectory[1]:F4}, {trajectory[2]:F4}) to ({invertedTrajectory[0]:F4}, {invertedTrajectory[1]:F4}, {invertedTrajectory[2]:F4})");
                            return invertedTrajectory;
                        }
                        if (trajectory != null && trajectory.Length >= 3) return (double[])trajectory.Clone();
                        return new double[] { 0, 0, 0 };
                    }

                    double[] productTrajectoryAdjusted = InvertTrajectoryIfNeeded(prodTrajectory);
                    double[] serviceTrajectoryAdjusted = InvertTrajectoryIfNeeded(servTrajectory);

                    void RecursivePlotTrajectory(double[] currentPosition, double[] trajectory, double magnitude, List<double[]> points, List<double> intensities, int depth, string trajectoryName)
                    {
                        if (currentPosition == null || trajectory == null || currentPosition.Length < 3 || trajectory.Length < 3) return;
                        if (depth >= MAX_RECURSION_DEPTH)
                        {
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unit B {trajectoryName} recursion stopped at max depth {depth}");
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unit B {trajectoryName} final position: ({currentPosition[0]:F6}, {currentPosition[1]:F6}, {currentPosition[2]:F4})");
                            points.Add((double[])currentPosition.Clone());
                            double finalPointIntensity = magnitude * Math.Pow(recursionFactor, depth);
                            intensities.Add(finalPointIntensity);
                            return;
                        }
                        if (currentPosition[0] < CONTINUE_PAST_PLANE && currentPosition[1] < CONTINUE_PAST_PLANE)
                        {
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unit B {trajectoryName} recursion stopped - Reached target negative threshold at depth {depth}");
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unit B {trajectoryName} final position: ({currentPosition[0]:F6}, {currentPosition[1]:F6}, {currentPosition[2]:F4})");
                            points.Add((double[])currentPosition.Clone());
                            double finalPointIntensity = magnitude * Math.Pow(recursionFactor, depth);
                            intensities.Add(finalPointIntensity);
                            return;
                        }
                        points.Add((double[])currentPosition.Clone());
                        double currentPointIntensity = magnitude * Math.Pow(recursionFactor, depth);
                        intensities.Add(currentPointIntensity);
                        bool beyondXPlane = currentPosition[0] < -1e-6;
                        bool beyondYPlane = currentPosition[1] < -1e-6;
                        bool beyondBothPlanes = beyondXPlane && beyondYPlane;
                        if (depth % 5 == 0 || beyondBothPlanes || (depth > 0 && points.Count > 1 && ((points[points.Count - 2][0] >= -1e-6 && beyondXPlane) || (points[points.Count - 2][1] >= -1e-6 && beyondYPlane))))
                        {
                            string positionInfo = "";
                            if (beyondXPlane) positionInfo += " BEYOND-X-PLANE";
                            if (beyondYPlane) positionInfo += " BEYOND-Y-PLANE";
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Unit B {trajectoryName} point {depth}: Position=({currentPosition[0]:F6}, {currentPosition[1]:F6}, {currentPosition[2]:F4}), Intensity={currentPointIntensity:F4}{positionInfo}");
                        }
                        double stepMultiplier = 1.0; // Different step logic for B
                        if (depth < 15) stepMultiplier = 2.5;
                        else if (!beyondBothPlanes && depth < MAX_RECURSION_DEPTH - 10) stepMultiplier = 2.0;
                        else stepMultiplier = 1.2;
                        double stepSize = magnitude * Math.Pow(recursionFactor, depth) * 0.3 * stepMultiplier; // Smaller base step for B
                        double[] nextPosition = new double[3];
                        for (int i = 0; i < 3; i++) nextPosition[i] = currentPosition[i] + trajectory[i] * stepSize;
                        RecursivePlotTrajectory(nextPosition, trajectory, magnitude, points, intensities, depth + 1, trajectoryName);
                    }

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Generating Product trajectory recursive plot (Unit B)");
                    RecursivePlotTrajectory(productCurrentPosition, productTrajectoryAdjusted, prodOverallMagnitude, productTrajectoryPoints, productPointIntensities, 0, "PRODUCT_B");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Generating Service trajectory recursive plot (Unit B)");
                    RecursivePlotTrajectory(serviceCurrentPosition, serviceTrajectoryAdjusted, servOverallMagnitude, serviceTrajectoryPoints, servicePointIntensities, 0, "SERVICE_B");

                    double[] productXPlaneIntersection = CalculatePlaneIntersection(productTrajectoryPoints, 0, 1e-6);
                    double[] productYPlaneIntersection = CalculatePlaneIntersection(productTrajectoryPoints, 1, 1e-6);
                    double[] serviceXPlaneIntersection = CalculatePlaneIntersection(serviceTrajectoryPoints, 0, 1e-6);
                    double[] serviceYPlaneIntersection = CalculatePlaneIntersection(serviceTrajectoryPoints, 1, 1e-6);

                    double[] productVector = (productTrajectoryAdjusted != null && productTrajectoryAdjusted.Length >= 3) ? (double[])productTrajectoryAdjusted.Clone() : new double[] { 0, 0, 0 };
                    double productVelocity = CalculateVelocity(productVector, prodOverallMagnitude);
                    double[] productPositiveCoordinate = FindPositiveCoordinate(productTrajectoryPoints, 1e-6);
                    double[] productNegativeCoordinate = FindNegativeCoordinate(productTrajectoryPoints, 1e-6);

                    double[] serviceVector = (serviceTrajectoryAdjusted != null && serviceTrajectoryAdjusted.Length >= 3) ? (double[])serviceTrajectoryAdjusted.Clone() : new double[] { 0, 0, 0 };
                    double serviceVelocity = CalculateVelocity(serviceVector, servOverallMagnitude);
                    double[] servicePositiveCoordinate = FindPositiveCoordinate(serviceTrajectoryPoints, 1e-6);
                    double[] serviceNegativeCoordinate = FindNegativeCoordinate(serviceTrajectoryPoints, 1e-6);

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] ----- PLANE INTERSECTION ANALYSIS (Unit B) -----");
                    if (productXPlaneIntersection != null) Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product X-Plane Intersection (Unit B): (0.000000, {productXPlaneIntersection[1]:F6}, {productXPlaneIntersection[2]:F6})"); else Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product trajectory (Unit B) does not intersect X-Plane");
                    if (productYPlaneIntersection != null) Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product Y-Plane Intersection (Unit B): ({productYPlaneIntersection[0]:F6}, 0.000000, {productYPlaneIntersection[2]:F6})"); else Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product trajectory (Unit B) does not intersect Y-Plane");
                    if (serviceXPlaneIntersection != null) Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service X-Plane Intersection (Unit B): (0.000000, {serviceXPlaneIntersection[1]:F6}, {serviceXPlaneIntersection[2]:F6})"); else Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service trajectory (Unit B) does not intersect X-Plane");
                    if (serviceYPlaneIntersection != null) Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service Y-Plane Intersection (Unit B): ({serviceYPlaneIntersection[0]:F6}, 0.000000, {serviceYPlaneIntersection[2]:F6})"); else Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service trajectory (Unit B) does not intersect Y-Plane");

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] ----- KEY TRAJECTORY DATA (Unit B) -----");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product Vector (Unit B): ({productVector[0]:F6}, {productVector[1]:F6}, {productVector[2]:F6})");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product Velocity (Unit B): {productVelocity:F6}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product Positive Coordinate (Unit B): ({productPositiveCoordinate[0]:F6}, {productPositiveCoordinate[1]:F6}, {productPositiveCoordinate[2]:F6})");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product Negative Coordinate (Unit B): ({productNegativeCoordinate[0]:F6}, {productNegativeCoordinate[1]:F6}, {productNegativeCoordinate[2]:F6})");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service Vector (Unit B): ({serviceVector[0]:F6}, {serviceVector[1]:F6}, {serviceVector[2]:F6})");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service Velocity (Unit B): {serviceVelocity:F6}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service Positive Coordinate (Unit B): ({servicePositiveCoordinate[0]:F6}, {servicePositiveCoordinate[1]:F6}, {servicePositiveCoordinate[2]:F6})");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service Negative Coordinate (Unit B): ({serviceNegativeCoordinate[0]:F6}, {serviceNegativeCoordinate[1]:F6}, {serviceNegativeCoordinate[2]:F6})");

                    int productNegativeXCount = CountNegativePoints(productTrajectoryPoints, 0, 1e-6);
                    int productNegativeYCount = CountNegativePoints(productTrajectoryPoints, 1, 1e-6);
                    int productNegativeBothCount = CountNegativeBothPoints(productTrajectoryPoints, 1e-6);
                    int serviceNegativeXCount = CountNegativePoints(serviceTrajectoryPoints, 0, 1e-6);
                    int serviceNegativeYCount = CountNegativePoints(serviceTrajectoryPoints, 1, 1e-6);
                    int serviceNegativeBothCount = CountNegativeBothPoints(serviceTrajectoryPoints, 1e-6);

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product negative X count (Unit B): {productNegativeXCount}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product negative Y count (Unit B): {productNegativeYCount}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product negative both count (Unit B): {productNegativeBothCount}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service negative X count (Unit B): {serviceNegativeXCount}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service negative Y count (Unit B): {serviceNegativeYCount}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service negative both count (Unit B): {serviceNegativeBothCount}");

                    unitResultsStore["Product_B_TrajectoryPoints"] = productTrajectoryPoints;
                    unitResultsStore["Product_B_PointIntensities"] = productPointIntensities;
                    unitResultsStore["Service_B_TrajectoryPoints"] = serviceTrajectoryPoints;
                    unitResultsStore["Service_B_PointIntensities"] = servicePointIntensities;
                    unitResultsStore["Product_B_XPlaneIntersection"] = productXPlaneIntersection;
                    unitResultsStore["Product_B_YPlaneIntersection"] = productYPlaneIntersection;
                    unitResultsStore["Service_B_XPlaneIntersection"] = serviceXPlaneIntersection;
                    unitResultsStore["Service_B_YPlaneIntersection"] = serviceYPlaneIntersection;
                    unitResultsStore["Product_B_Vector"] = productVector;
                    unitResultsStore["Product_B_Velocity"] = productVelocity;
                    unitResultsStore["Product_B_PositiveCoordinate"] = productPositiveCoordinate;
                    unitResultsStore["Product_B_NegativeCoordinate"] = productNegativeCoordinate;
                    unitResultsStore["Service_B_Vector"] = serviceVector;
                    unitResultsStore["Service_B_Velocity"] = serviceVelocity;
                    unitResultsStore["Service_B_PositiveCoordinate"] = servicePositiveCoordinate;
                    unitResultsStore["Service_B_NegativeCoordinate"] = serviceNegativeCoordinate;
                    unitResultsStore["Product_B_NegativeXCount"] = productNegativeXCount;
                    unitResultsStore["Product_B_NegativeYCount"] = productNegativeYCount;
                    unitResultsStore["Product_B_NegativeBothCount"] = productNegativeBothCount;
                    unitResultsStore["Service_B_NegativeXCount"] = serviceNegativeXCount;
                    unitResultsStore["Service_B_NegativeYCount"] = serviceNegativeYCount;
                    unitResultsStore["Service_B_NegativeBothCount"] = serviceNegativeBothCount;

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Product trajectory plot (Unit B): {productTrajectoryPoints?.Count ?? 0} points, {productNegativeBothCount} in negative X-Y quadrant");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Service trajectory plot (Unit B): {serviceTrajectoryPoints?.Count ?? 0} points, {serviceNegativeBothCount} in negative X-Y quadrant");

                    string result = $"FeatureTensorsAndMapping_B_Cust_{custId}_BasedOn_{analysisResult.Replace("InitialAnalysis_B_", "")}";
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 2 (Unit B) - Feature tensor generation and mapping completed: {result}");
                    return result;
                }

                //==========================================================================
                // Step 3 (Unit B): Processed Feature Definition Creation
                //==========================================================================
                string Stage3_ProcessedFeatureDefinition_B(string tensorMappingResult, int custId)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 3 (Unit B) - Creating processed feature definition for customer {custId}.");

                    double productVelocity = unitResultsStore.TryGetValue("Product_B_Velocity", out var pv) ? Convert.ToDouble(pv) : 0.5;
                    double serviceVelocity = unitResultsStore.TryGetValue("Service_B_Velocity", out var sv) ? Convert.ToDouble(sv) : 0.5;
                    double[] productXPlaneIntersection = unitResultsStore.TryGetValue("Product_B_XPlaneIntersection", out var pxi) ? pxi as double[] : null;
                    double[] productYPlaneIntersection = unitResultsStore.TryGetValue("Product_B_YPlaneIntersection", out var pyi) ? pyi as double[] : null;
                    double[] serviceXPlaneIntersection = unitResultsStore.TryGetValue("Service_B_XPlaneIntersection", out var sxi) ? sxi as double[] : null;
                    double[] serviceYPlaneIntersection = unitResultsStore.TryGetValue("Service_B_YPlaneIntersection", out var syi) ? syi as double[] : null;
                    int productNegativeBothCount = unitResultsStore.TryGetValue("Product_B_NegativeBothCount", out var pnbc) ? Convert.ToInt32(pnbc) : 0;
                    int serviceNegativeBothCount = unitResultsStore.TryGetValue("Service_B_NegativeBothCount", out var snbc) ? Convert.ToInt32(snbc) : 0;

                    string processingLevel = "StandardB";
                    if (productVelocity > 0.9 || serviceVelocity > 0.9) processingLevel = "PremiumB";
                    else if (productVelocity > 0.6 || serviceVelocity > 0.6) processingLevel = "EnhancedB";

                    string processingModifier = "";
                    if (productNegativeBothCount > 5 && serviceNegativeBothCount > 5) processingModifier = "DeepNegativeB";
                    else if (productXPlaneIntersection != null && serviceXPlaneIntersection != null && productYPlaneIntersection != null && serviceYPlaneIntersection != null)
                    {
                        double xPlaneDistanceY = Math.Abs(productXPlaneIntersection[1] - serviceXPlaneIntersection[1]);
                        double xPlaneDistanceZ = Math.Abs(productXPlaneIntersection[2] - serviceXPlaneIntersection[2]);
                        double yPlaneDistanceX = Math.Abs(productYPlaneIntersection[0] - serviceYPlaneIntersection[0]);
                        double yPlaneDistanceZ = Math.Abs(productYPlaneIntersection[2] - serviceYPlaneIntersection[2]);
                        double avgIntersectionDistance = (xPlaneDistanceY + xPlaneDistanceZ + yPlaneDistanceX + yPlaneDistanceZ) / 4.0;
                        if (avgIntersectionDistance < 0.15) processingModifier = "ConvergentB";
                        else if (avgIntersectionDistance < 0.4) processingModifier = "AlignedB";
                        else processingModifier = "DivergentB";
                    }
                    else if (productNegativeBothCount > 5 || serviceNegativeBothCount > 5)
                    {
                        processingModifier = productNegativeBothCount > serviceNegativeBothCount ? "ProductNegativeDominantB" : "ServiceNegativeDominantB";
                    }

                    string result = $"ProcessedFeatures_B_Cust_{custId}_Level_{processingLevel}";
                    if (!string.IsNullOrEmpty(processingModifier)) result += $"_{processingModifier}";
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 3 (Unit B) - Processed feature definition created: {result}");
                    return result;
                }

                //==========================================================================
                // Step 4 (Unit B): Feature Quality Assessment
                //==========================================================================
                string Stage4_FeatureQualityAssessment_B(string processedFeatureResult, int custId)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 4 (Unit B) - Assessing feature quality for customer {custId}.");

                    double productVelocity = unitResultsStore.TryGetValue("Product_B_Velocity", out var pv) ? Convert.ToDouble(pv) : 0.5;
                    double serviceVelocity = unitResultsStore.TryGetValue("Service_B_Velocity", out var sv) ? Convert.ToDouble(sv) : 0.5;
                    var productTrajectoryPoints = unitResultsStore.TryGetValue("Product_B_TrajectoryPoints", out var ptp) ? ptp as List<double[]> : new List<double[]>();
                    // var serviceTrajectoryPoints = unitResultsStore.TryGetValue("Service_B_TrajectoryPoints", out var stp) ? stp as List<double[]> : new List<double[]>(); // Not used
                    double[] productXPlaneIntersection = unitResultsStore.TryGetValue("Product_B_XPlaneIntersection", out var pxi) ? pxi as double[] : null;
                    double[] productYPlaneIntersection = unitResultsStore.TryGetValue("Product_B_YPlaneIntersection", out var pyi) ? pyi as double[] : null;

                    double velocityComponent = (productVelocity + serviceVelocity) / 2.0;
                    double trajectoryStability = 0.5;
                    double intersectionQuality = 0.5;

                    if (productTrajectoryPoints != null && productTrajectoryPoints.Count > 1)
                    {
                        trajectoryStability = CalculateTrajectoryStability(productTrajectoryPoints);
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] QA product trajectory stability (Unit B): {trajectoryStability:F4}");
                    }
                    if (productXPlaneIntersection != null && productYPlaneIntersection != null)
                    {
                        double zDifference = Math.Abs(productXPlaneIntersection[2] - productYPlaneIntersection[2]);
                        intersectionQuality = 1.0 - Math.Min(1.0, zDifference);
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] QA intersection quality (Unit B): {intersectionQuality:F4}");
                    }

                    double qaScore = velocityComponent * 0.35 + trajectoryStability * 0.35 + intersectionQuality * 0.30; // Different weights for B
                    qaScore = Math.Min(qaScore, 1.0);
                    int qaLevel = (int)(qaScore * 4) + 1; // Different level scaling for B
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] QA final score (Unit B): {qaScore:F4}, level: {qaLevel}");
                    string result = $"QualityAssessment_B_Passed_Level_{qaLevel}_V{velocityComponent:F2}_S{trajectoryStability:F2}_I{intersectionQuality:F2}";
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 4 (Unit B) - Feature quality assessment completed: {result}");
                    return result;
                }


                //==========================================================================
                // Step 5 (Unit B): Combined Feature Evaluation
                //==========================================================================
                float Stage5_CombinedFeatureEvaluation_B(string qualityAssessmentResult, int custId)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 5 (Unit B) - Evaluating combined features for customer {custId}.");

                    double productVelocity = unitResultsStore.TryGetValue("Product_B_Velocity", out var pv) ? Convert.ToDouble(pv) : 0.5;
                    double serviceVelocity = unitResultsStore.TryGetValue("Service_B_Velocity", out var sv) ? Convert.ToDouble(sv) : 0.5;
                    int productNegativeBothCount = unitResultsStore.TryGetValue("Product_B_NegativeBothCount", out var pnbc) ? Convert.ToInt32(pnbc) : 0;
                    int serviceNegativeBothCount = unitResultsStore.TryGetValue("Service_B_NegativeBothCount", out var snbc) ? Convert.ToInt32(snbc) : 0;
                    int totalNegativePoints = productNegativeBothCount + serviceNegativeBothCount;
                    double[] productVector = unitResultsStore.TryGetValue("Product_B_Vector", out var pvec) ? pvec as double[] : new double[] { 0, 0, 0 };
                    double[] serviceVector = unitResultsStore.TryGetValue("Service_B_Vector", out var svec) ? svec as double[] : new double[] { 0, 0, 0 };

                    double alignmentScore = 0.5;
                    double productMagSq = productVector != null ? productVector[0] * productVector[0] + productVector[1] * productVector[1] + productVector[2] * productVector[2] : 0;
                    double serviceMagSq = serviceVector != null ? serviceVector[0] * serviceVector[0] + serviceVector[1] * serviceVector[1] + serviceVector[2] * serviceVector[2] : 0;
                    double productMag = Math.Sqrt(productMagSq);
                    double serviceMag = Math.Sqrt(serviceMagSq);

                    if (productMag > 1e-9 && serviceMag > 1e-9)
                    {
                        double dotProduct = 0;
                        if (productVector != null && serviceVector != null)
                        {
                            for (int i = 0; i < Math.Min(productVector.Length, serviceVector.Length); i++)
                            {
                                dotProduct += productVector[i] * serviceVector[i];
                            }
                        }
                        alignmentScore = dotProduct / (productMag * serviceMag);
                        alignmentScore = Math.Max(-1.0, Math.Min(1.0, alignmentScore));
                        alignmentScore = (alignmentScore + 1.0) / 2.0;
                    }

                    float baseScore = 0.65f + (custId % 12) / 15.0f; // Different base for B
                    float velocityBonus = (float)((productVelocity + serviceVelocity) / 3.5); // Different velocity bonus scaling
                    float alignmentBonus = (float)(alignmentScore / 4); // Different alignment bonus scaling
                    float negativeBonus = (float)(Math.Min(totalNegativePoints, 15) / 40.0); // Different negative bonus scaling
                    float result = Math.Min(baseScore + velocityBonus + alignmentBonus + negativeBonus, 1.0f);

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 5 (Unit B) - Combined feature evaluation calculation.");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Base Score: {baseScore:F4}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Velocity Bonus: {velocityBonus:F4} (Product B: {productVelocity:F4}, Service B: {serviceVelocity:F4})");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Alignment Bonus: {alignmentBonus:F4} (Alignment Score B: {alignmentScore:F4})");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Negative Trajectory Bonus (Unit B): {negativeBonus:F4} (Total Negative Points B: {totalNegativePoints})");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Final Score (Unit B): {result:F4}");
                    return result;
                }


                //==========================================================================
                // Step 6 (Unit B): Fractal Optimization Analysis
                //==========================================================================
                string Stage6_FractalOptimizationAnalysis_B(string evaluationResult, float evaluationScore, int custId)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 6 (Unit B) - Performing fractal optimization analysis for customer {custId}.");

                    double productVelocity = unitResultsStore.TryGetValue("Product_B_Velocity", out var pv) ? Convert.ToDouble(pv) : 0.5;
                    double serviceVelocity = unitResultsStore.TryGetValue("Service_B_Velocity", out var sv) ? Convert.ToDouble(sv) : 0.5;
                    double[] productXPlaneIntersection = unitResultsStore.TryGetValue("Product_B_XPlaneIntersection", out var pxi) ? pxi as double[] : null;
                    double[] productYPlaneIntersection = unitResultsStore.TryGetValue("Product_B_YPlaneIntersection", out var pyi) ? pyi as double[] : null;
                    double[] serviceXPlaneIntersection = unitResultsStore.TryGetValue("Service_B_XPlaneIntersection", out var sxi) ? sxi as double[] : null;
                    double[] serviceYPlaneIntersection = unitResultsStore.TryGetValue("Service_B_YPlaneIntersection", out var syi) ? syi as double[] : null;

                    Console.WriteLine("========== PRODUCT INTERSECTIONS (Unit B) ==========");
                    if (productXPlaneIntersection != null) Console.WriteLine($"Product X-Plane Intersection (Unit B): (0.0, {productXPlaneIntersection[1]:F6}, {productXPlaneIntersection[2]:F6})"); else Console.WriteLine("Product X-Plane Intersection (Unit B): null");
                    if (productYPlaneIntersection != null) Console.WriteLine($"Product Y-Plane Intersection (Unit B): ({productYPlaneIntersection[0]:F6}, 0.0, {productYPlaneIntersection[2]:F6})"); else Console.WriteLine("Product Y-Plane Intersection (Unit B): null");
                    Console.WriteLine("========== SERVICE INTERSECTIONS (Unit B) ==========");
                    if (serviceXPlaneIntersection != null) Console.WriteLine($"Service X-Plane Intersection (Unit B): (0.0, {serviceXPlaneIntersection[1]:F6}, {serviceXPlaneIntersection[2]:F6})"); else Console.WriteLine("Service X-Plane Intersection (Unit B): null");
                    if (serviceYPlaneIntersection != null) Console.WriteLine($"Service Y-Plane Intersection (Unit B): ({serviceYPlaneIntersection[0]:F6}, 0.0, {serviceYPlaneIntersection[2]:F6})"); else Console.WriteLine("Service Y-Plane Intersection (Unit B): null");

                    const int Power = 6; // Different Power for B
                    const float EscapeThreshold = 2.5f; // Different Threshold for B
                    const int MaxIterations = 40; // Different Iterations for B

                    float productXPlaneVelocity = (float)(productXPlaneIntersection != null ? productVelocity : 0.0);
                    float productYPlaneVelocity = (float)(productYPlaneIntersection != null ? productVelocity : 0.0);
                    float serviceXPlaneVelocity = (float)(serviceXPlaneIntersection != null ? serviceVelocity : 0.0);
                    float serviceYPlaneVelocity = (float)(serviceYPlaneIntersection != null ? serviceVelocity : 0.0);

                    Console.WriteLine("========== INTERSECTION VELOCITIES (Unit B) ==========");
                    Console.WriteLine($"Product X-Plane Velocity (Unit B): {productXPlaneVelocity:F4}");
                    Console.WriteLine($"Product Y-Plane Velocity (Unit B): {productYPlaneVelocity:F4}");
                    Console.WriteLine($"Service X-Plane Velocity (Unit B): {serviceXPlaneVelocity:F4}");
                    Console.WriteLine($"Service Y-Plane Velocity (Unit B): {serviceYPlaneVelocity:F4}");

                    List<(System.Numerics.Vector3 position, float velocity, string source)> velocitySources = new List<(System.Numerics.Vector3, float, string)>();
                    if (productXPlaneIntersection != null && productXPlaneIntersection.Length >= 3) velocitySources.Add((new System.Numerics.Vector3(0.0f, (float)productXPlaneIntersection[1], (float)productXPlaneIntersection[2]), productXPlaneVelocity * 1.1f, "ProductX_B"));
                    if (productYPlaneIntersection != null && productYPlaneIntersection.Length >= 3) velocitySources.Add((new System.Numerics.Vector3((float)productYPlaneIntersection[0], 0.0f, (float)productYPlaneIntersection[2]), productYPlaneVelocity * 1.1f, "ProductY_B"));
                    if (serviceXPlaneIntersection != null && serviceXPlaneIntersection.Length >= 3) velocitySources.Add((new System.Numerics.Vector3(0.0f, (float)serviceXPlaneIntersection[1], (float)serviceXPlaneIntersection[2]), serviceXPlaneVelocity * 1.1f, "ServiceX_B"));
                    if (serviceYPlaneIntersection != null && serviceYPlaneIntersection.Length >= 3) velocitySources.Add((new System.Numerics.Vector3((float)serviceYPlaneIntersection[0], 0.0f, (float)serviceYPlaneIntersection[2]), serviceYPlaneVelocity * 1.1f, "ServiceY_B"));

                    Console.WriteLine("========== VELOCITY SOURCES (Unit B) ==========");
                    foreach (var source in velocitySources) Console.WriteLine($"{source.source} Source Position (Unit B): ({source.position.X:F4}, {source.position.Y:F4}, {source.position.Z:F4}), Velocity: {source.velocity:F4}");

                    System.Numerics.Vector3[] samplePoints = new System.Numerics.Vector3[6]; // 6 samples for B
                    samplePoints[0] = (productXPlaneIntersection != null && productXPlaneIntersection.Length >= 3) ? new System.Numerics.Vector3(-0.05f, (float)productXPlaneIntersection[1], (float)productXPlaneIntersection[2]) : new System.Numerics.Vector3(-0.05f, 0.1f, 0.1f);
                    samplePoints[1] = (productYPlaneIntersection != null && productYPlaneIntersection.Length >= 3) ? new System.Numerics.Vector3((float)productYPlaneIntersection[0], -0.05f, (float)productYPlaneIntersection[2]) : new System.Numerics.Vector3(0.5f, -0.05f, 0.0f);
                    samplePoints[2] = (serviceXPlaneIntersection != null && serviceXPlaneIntersection.Length >= 3) ? new System.Numerics.Vector3(-0.05f, (float)serviceXPlaneIntersection[1], (float)serviceXPlaneIntersection[2]) : new System.Numerics.Vector3(-0.05f, 0.8f, 0.0f);
                    samplePoints[3] = (serviceYPlaneIntersection != null && serviceYPlaneIntersection.Length >= 3) ? new System.Numerics.Vector3((float)serviceYPlaneIntersection[0], -0.05f, (float)serviceYPlaneIntersection[2]) : new System.Numerics.Vector3(0.3f, -0.05f, 0.3f);
                    if (velocitySources.Count > 0) { System.Numerics.Vector3 sum = System.Numerics.Vector3.Zero; foreach (var source in velocitySources) sum += source.position; samplePoints[4] = sum / velocitySources.Count; } else { samplePoints[4] = new System.Numerics.Vector3(1.0f, 1.0f, 1.0f); }
                    samplePoints[5] = new System.Numerics.Vector3(0f, 0f, 0f); // Additional center point for B

                    Console.WriteLine("========== SAMPLE POINTS (Unit B) ==========");
                    for (int i = 0; i < samplePoints.Length; i++) Console.WriteLine($"Sample {i + 1} Coordinates (Unit B): ({samplePoints[i].X:F4}, {samplePoints[i].Y:F4}, {samplePoints[i].Z:F4})");

                    System.Numerics.Vector3[] sampleValues = new System.Numerics.Vector3[samplePoints.Length];
                    int[] sampleIterations = new int[samplePoints.Length];
                    float[] sampleVelocities = new float[samplePoints.Length];
                    Dictionary<int, Dictionary<string, float>> sampleContributions = new Dictionary<int, Dictionary<string, float>>();
                    for (int i = 0; i < samplePoints.Length; i++) { sampleContributions[i] = new Dictionary<string, float>(); foreach (var source in velocitySources) sampleContributions[i][source.source] = 0.0f; }

                    for (int sampleIndex = 0; sampleIndex < samplePoints.Length; sampleIndex++)
                    {
                        System.Numerics.Vector3 c = samplePoints[sampleIndex];
                        System.Numerics.Vector3 z = System.Numerics.Vector3.Zero;
                        int iterations = 0;
                        float diffusedVelocity = 0.0f;
                        Console.WriteLine($"========== PROCESSING SAMPLE {sampleIndex + 1} (Unit B) ==========");
                        Console.WriteLine($"Starting point (Unit B): ({c.X:F4}, {c.Y:F4}, {c.Z:F4})");
                        for (iterations = 0; iterations < MaxIterations; iterations++)
                        {
                            float rSq = z.LengthSquared();
                            if (rSq > EscapeThreshold * EscapeThreshold) { Console.WriteLine($"Unit B Escaped at iteration {iterations + 1}"); break; }
                            float r = MathF.Sqrt(rSq);
                            Console.WriteLine($"Unit B Iteration {iterations + 1}, z=({z.X:F6}, {z.Y:F6}, {z.Z:F6}), r={r:F6}");
                            foreach (var source in velocitySources)
                            {
                                float distanceSq = System.Numerics.Vector3.DistanceSquared(z, source.position);
                                float distance = MathF.Sqrt(distanceSq);
                                if (distance < 3.0f) // Different interaction distance for B
                                {
                                    float contribution = source.velocity * MathF.Exp(-distance * 1.8f) * MathF.Exp(-iterations * 0.08f); // Different diffusion params for B
                                    diffusedVelocity += contribution;
                                    if (sampleContributions[sampleIndex].ContainsKey(source.source)) sampleContributions[sampleIndex][source.source] += contribution; else sampleContributions[sampleIndex][source.source] = contribution;
                                    Console.WriteLine($"  Unit B Contribution from {source.source}: {contribution:F6} (distance: {distance:F4})");
                                }
                            }
                            float theta = (r < 1e-6f) ? 0 : MathF.Acos(z.Z / r);
                            float phi = MathF.Atan2(z.Y, z.X);
                            float newR = MathF.Pow(r, Power);
                            float newTheta = Power * theta;
                            float newPhi = Power * phi;
                            z = new System.Numerics.Vector3(newR * MathF.Sin(newTheta) * MathF.Cos(newPhi), newR * MathF.Sin(newTheta) * MathF.Sin(newPhi), newR * MathF.Cos(newTheta)) + c;
                        }
                        sampleValues[sampleIndex] = z;
                        sampleIterations[sampleIndex] = iterations;
                        sampleVelocities[sampleIndex] = diffusedVelocity;
                        Console.WriteLine($"Final Sample {sampleIndex + 1} Results (Unit B):");
                        Console.WriteLine($"  Final z value (Unit B): ({z.X:F6}, {z.Y:F6}, {z.Z:F6})");
                        Console.WriteLine($"  Iterations (Unit B): {iterations}");
                        Console.WriteLine($"  Total diffused velocity (Unit B): {diffusedVelocity:F6}");
                        Console.WriteLine($"  Contributions breakdown (Unit B):");
                        foreach (var source in velocitySources) if (sampleContributions[sampleIndex].ContainsKey(source.source)) Console.WriteLine($"    {source.source}: {sampleContributions[sampleIndex][source.source]:F6}");
                    }

                    unitResultsStore["ProductXPlaneVelocity_B"] = productXPlaneVelocity;
                    unitResultsStore["ProductYPlaneVelocity_B"] = productYPlaneVelocity;
                    unitResultsStore["ServiceXPlaneVelocity_B"] = serviceXPlaneVelocity;
                    unitResultsStore["ServiceYPlaneVelocity_B"] = serviceYPlaneVelocity;
                    for (int i = 0; i < samplePoints.Length; i++)
                    {
                        unitResultsStore[$"Sample{i + 1}Coordinate_B"] = samplePoints[i];
                        unitResultsStore[$"Sample{i + 1}Value_B"] = sampleValues[i];
                        unitResultsStore[$"Sample{i + 1}Iterations_B"] = sampleIterations[i];
                        unitResultsStore[$"Sample{i + 1}Velocity_B"] = sampleVelocities[i];
                        foreach (var source in velocitySources)
                        {
                            if (sampleContributions[i].ContainsKey(source.source)) unitResultsStore[$"Sample{i + 1}_{source.source}Contribution_B"] = sampleContributions[i][source.source];
                            else unitResultsStore[$"Sample{i + 1}_{source.source}Contribution_B"] = 0.0f;
                        }
                    }

                    System.Text.StringBuilder resultBuilder = new System.Text.StringBuilder();
                    resultBuilder.Append($"OptimizationAnalysis_B_Cust_{custId}");
                    resultBuilder.Append("_V[");
                    bool firstVelocity = true;
                    if (productXPlaneVelocity > 1e-6f || productXPlaneIntersection != null) { resultBuilder.Append($"PX_B:{productXPlaneVelocity:F3}"); firstVelocity = false; }
                    if (productYPlaneVelocity > 1e-6f || productYPlaneIntersection != null) { if (!firstVelocity) resultBuilder.Append(","); resultBuilder.Append($"PY_B:{productYPlaneVelocity:F3}"); firstVelocity = false; }
                    if (serviceXPlaneVelocity > 1e-6f || serviceXPlaneIntersection != null) { if (!firstVelocity) resultBuilder.Append(","); resultBuilder.Append($"SX_B:{serviceXPlaneVelocity:F3}"); firstVelocity = false; }
                    if (serviceYPlaneVelocity > 1e-6f || serviceYPlaneIntersection != null) { if (!firstVelocity) resultBuilder.Append(","); resultBuilder.Append($"SY_B:{serviceYPlaneVelocity:F3}"); }
                    resultBuilder.Append("]");
                    resultBuilder.Append("_S[");
                    for (int i = 0; i < samplePoints.Length; i++) { string status = sampleIterations[i] >= MaxIterations ? "InSet" : $"Escaped({sampleIterations[i]})"; resultBuilder.Append($"P{i + 1}_B:{sampleVelocities[i]:F4}_S{status}"); if (i < samplePoints.Length - 1) resultBuilder.Append(","); }
                    resultBuilder.Append("]");
                    string result = resultBuilder.ToString();
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 6 (Unit B) - Fractal optimization analysis completed: {result}");
                    return result;
                }


                //==========================================================================
                // Step 7 (Unit B): Tensor Network Training
                //==========================================================================

                string Stage7_TensorNetworkTraining_B(string optimizationResult, int custId, ConcurrentDictionary<string, object> resultsStore)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 7 (Unit B) - Training tensor network for customer {custId} using Actual TF.NET Model B.");
                    tf.compat.v1.disable_eager_execution();
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Disabled eager execution for TensorFlow operations for Unit B.");

                    byte[]? modelWeightsBytes = RuntimeProcessingContext.RetrieveContextValue("SequentialProcessingUnitC_SerializedModelData") as byte[];
                    byte[]? modelBiasBytes = RuntimeProcessingContext.RetrieveContextValue("SequentialProcessingUnitC_AncillaryData") as byte[];
                    float[] eigenvalues = unitResultsStore.TryGetValue("B_MarketCurvatureEigenvalues", out var eigVals) && eigVals is float[] eigArray ? eigArray : new float[] { 1.0f, 1.0f, 1.0f };
                    resultsStore["B_MarketCurvatureEigenvalues"] = eigenvalues;

                    int numEpochs = 80; // Different epochs for B
                    List<float> trainingLosses = new List<float>();
                    List<float> trainingErrors = new List<float>();

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Step 7 (Unit B) - Creating sample training data.");
                    float[][] numericalSamples = new float[][] { // Slightly different data for B
                        new float[] { 0.35f, 0.75f, 0.15f, 0.80f }, new float[] { 0.55f, 0.25f, 0.95f, 0.30f }, new float[] { 0.85f, 0.65f, 0.45f, 0.50f }, new float[] { 0.15f, 0.85f, 0.65f, 0.20f },
                        new float[] { 0.75f, 0.35f, 0.25f, 0.90f }, new float[] { 0.45f, 0.55f, 0.75f, 0.60f }, new float[] { 0.25f, 0.95f, 0.35f, 0.10f }, new float[] { 0.65f, 0.15f, 0.85f, 0.70f },
                        new float[] { 0.30f, 0.60f, 0.10f, 0.85f }, new float[] { 0.50f, 0.20f, 0.80f, 0.35f }, new float[] { 0.70f, 0.50f, 0.40f, 0.55f }, new float[] { 0.10f, 0.70f, 0.50f, 0.25f },
                        new float[] { 0.60f, 0.30f, 0.20f, 0.95f }, new float[] { 0.40f, 0.40f, 0.60f, 0.65f }, new float[] { 0.20f, 0.80f, 0.30f, 0.15f }, new float[] { 0.55f, 0.10f, 0.70f, 0.75f }
                    };
                    float[] numericalLabels = new float[numericalSamples.Length];
                    for (int i = 0; i < numericalSamples.Length; i++)
                    {
                        if (numericalSamples[i] == null || numericalSamples[i].Length < 4) { numericalLabels[i] = 0.0f; continue; }
                        float x = numericalSamples[i][0], y = numericalSamples[i][1], z = numericalSamples[i][2], p = numericalSamples[i][3];
                        numericalLabels[i] = x * 2.0f * (float)Math.Sin(p) + y * 2.0f * (float)Math.Cos(p) + z * 2.0f * (float)Math.Sin(p / 2f) + x * y * z * 0.2f; // Different target function for B
                    }
                    string[] wordSamples = new string[] { // Different word samples for B
                        "strategic alignment high", "operational synergy excellent", "technology adoption rapid", "regulatory compliance strong", "market share increasing", "supply chain optimized",
                        "financial performance solid", "risk management effective", "customer engagement active", "employee productivity high", "innovation pipeline robust", "distribution network wide",
                        "environmental impact low", "social responsibility high", "governance structure sound", "competitive landscape favorable"
                    };
                    float[][] wordEmbeddings = TransformWordsToEmbeddings(wordSamples);
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Created {numericalSamples.Length} numerical samples and {wordSamples.Length} word-based samples (Unit B).");

                    float[,] numericalData = ConvertJaggedToMultidimensional(numericalSamples);
                    float[,] wordData = ConvertJaggedToMultidimensional(wordEmbeddings);
                    float[,] targetValues = new float[numericalLabels.Length, 1];
                    for (int i = 0; i < numericalLabels.Length; i++) targetValues[i, 0] = numericalLabels[i];

                    int numericalFeatureCount = numericalData.GetLength(1);
                    int wordFeatureCount = wordData.GetLength(1);
                    int totalInputFeatureCount = numericalFeatureCount + wordFeatureCount;

                    int batchSize = 6; // Different batch size for B
                    int dataSize = numericalData.GetLength(0);
                    int actualBatchSize = Math.Min(batchSize, dataSize);
                    if (actualBatchSize <= 0 && dataSize > 0) actualBatchSize = dataSize;
                    else if (dataSize == 0)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning (Unit B): No training data available. Skipping training.");
                        resultsStore["ModelBProcessingWarning"] = "No training data available.";
                        resultsStore["ModelBProcessingOutcome"] = 0.1f;
                        resultsStore["ModelBTrainingError"] = float.NaN;
                        return $"TensorNetworkTrainingSkipped_B_Cust_{custId}_NoData";
                    }
                    if (actualBatchSize <= 0) actualBatchSize = 1;
                    int numBatches = (actualBatchSize > 0) ? (int)Math.Ceiling((double)dataSize / actualBatchSize) : 0;
                    int[] indices = Enumerable.Range(0, dataSize).ToArray();

                    string initialExpression = "2*2"; // Different expression for B
                    string regexPattern = ConvertExpressionToRegex(initialExpression);
                    string nDimensionalExpression = ConvertRegexToNDimensionalExpression(regexPattern);

                    if (modelWeightsBytes != null && modelBiasBytes != null && modelWeightsBytes.Length > 0 && modelBiasBytes.Length > 0)
                    {
                        modelBGraph = tf.Graph(); // Initialize graph for Model B
                        modelBGraph.as_default(); // Set as default graph for operation creation
                        {
                            modelBSession = tf.Session(modelBGraph); // Create session with Model B's graph
                            try
                            {
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Step 7 (Unit B) - Initializing Model B Architecture in its own graph.");
                                float[] unitCWeightsArray = DeserializeFloatArray(modelWeightsBytes);
                                float[] unitCBiasArray = DeserializeFloatArray(modelBiasBytes);
                                int unitCHiddenSize = -1;
                                if (unitCBiasArray.Length >= 1) unitCHiddenSize = unitCBiasArray.Length - 1;
                                if (unitCHiddenSize <= 0 || unitCWeightsArray.Length != (totalInputFeatureCount * unitCHiddenSize) + (unitCHiddenSize * 1))
                                {
                                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Warning (Unit B): Could not reliably infer Unit C hidden size. Using fallback.");
                                    unitCHiddenSize = 64;
                                }
                                int hiddenLayerSizeB = 70; // Different hidden layer size for B
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Model B architecture parameters: Input Feats: {totalInputFeatureCount}, Hidden Size: {hiddenLayerSizeB}");

                                float[,] modelBWeights1Data = GenerateWeightsFromExpression(nDimensionalExpression, totalInputFeatureCount, hiddenLayerSizeB);
                                float[,] modelBWeights2Data = GenerateWeightsFromExpression(nDimensionalExpression, hiddenLayerSizeB, 1);

                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Defining TensorFlow operations for Model B.");
                                Tensor numericalInput_B_Ops = tf.placeholder(tf.float32, shape: new int[] { -1, numericalFeatureCount }, name: "numerical_input_B");
                                Tensor wordInput_B_Ops = tf.placeholder(tf.float32, shape: new int[] { -1, wordFeatureCount }, name: "word_input_B");
                                Tensor targetOutput_B_Ops = tf.placeholder(tf.float32, shape: new int[] { -1, 1 }, name: "target_output_B");
                                Tensor combinedInput_B_Ops = tf.concat(new[] { numericalInput_B_Ops, wordInput_B_Ops }, axis: 1, name: "combined_input_B");
                                ResourceVariable weights1_B_Ops = tf.Variable(tf.constant(modelBWeights1Data, dtype: tf.float32), name: "weights1_B");
                                ResourceVariable bias1_B_Ops = tf.Variable(tf.zeros(hiddenLayerSizeB, dtype: tf.float32), name: "bias1_B");
                                ResourceVariable weights2_B_Ops = tf.Variable(tf.constant(modelBWeights2Data, dtype: tf.float32), name: "weights2_B");
                                ResourceVariable bias2_B_Ops = tf.Variable(tf.zeros(1, dtype: tf.float32), name: "bias2_B");
                                Tensor hidden_B_Ops = tf.nn.sigmoid(tf.add(tf.matmul(combinedInput_B_Ops, weights1_B_Ops), bias1_B_Ops), name: "hidden_B"); // Sigmoid for B
                                Tensor predictions_B_Ops = tf.add(tf.matmul(hidden_B_Ops, weights2_B_Ops), bias2_B_Ops, name: "predictions_B");
                                Tensor loss_B_Ops = tf.reduce_mean(tf.square(tf.subtract(predictions_B_Ops, targetOutput_B_Ops)), name: "mse_loss_B");
                                var regularizer_B = tf.reduce_sum(tf.square(weights1_B_Ops)) / 2.0f + tf.reduce_sum(tf.square(weights2_B_Ops)) / 2.0f; // L2 Regularization for B
                                Tensor lossWithRegularization_B_Ops = loss_B_Ops + 0.0005f * regularizer_B; // Add regularization to loss for B
                                var optimizer_B = tf.train.AdamOptimizer(0.001f);
                                Operation trainOp_B_Ops = optimizer_B.minimize(lossWithRegularization_B_Ops);
                                Tensor meanAbsError_B_Ops = tf.reduce_mean(tf.abs(tf.subtract(predictions_B_Ops, targetOutput_B_Ops)), name: "mae_B");
                                Operation initOp_B_Ops = tf.global_variables_initializer();
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] TensorFlow operations defined for Unit B.");

                                modelBSession.run(initOp_B_Ops); // Initialize variables within the correct session
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Model B - Actual TensorFlow.NET variables initialized in its own session.");

                                var trainingDataFeed_B = new FeedItem[] {
                                    new FeedItem(numericalInput_B_Ops, numericalData), new FeedItem(wordInput_B_Ops, wordData), new FeedItem(targetOutput_B_Ops, targetValues)
                                };

                                for (int epoch = 0; epoch < numEpochs; epoch++)
                                {
                                    ShuffleArray(indices); float epochLoss = 0.0f;
                                    for (int batchLoop = 0; batchLoop < numBatches; batchLoop++) // Changed loop variable name
                                    {
                                        int startIdx = batchLoop * actualBatchSize; int endIdx = Math.Min(startIdx + actualBatchSize, dataSize); int batchCount = endIdx - startIdx; if (batchCount <= 0) continue;
                                        float[,] batchNumerical = ExtractBatch(numericalData, indices, startIdx, batchCount); float[,] batchWord = ExtractBatch(wordData, indices, startIdx, batchCount); float[,] batchTarget = ExtractBatch(targetValues, indices, startIdx, batchCount);
                                        var batchFeed_B = new FeedItem[] {
                                            new FeedItem(numericalInput_B_Ops, batchNumerical), new FeedItem(wordInput_B_Ops, batchWord), new FeedItem(targetOutput_B_Ops, batchTarget)
                                        };
                                        var results_B = modelBSession.run(new ITensorOrOperation[] { lossWithRegularization_B_Ops, trainOp_B_Ops }, batchFeed_B); // Use modelBSession
                                        float batchLossValue = ((Tensor)results_B[0]).numpy().ToArray<float>()[0];
                                        epochLoss += batchLossValue;
                                        if (batchLoop % 5 == 0 || batchLoop == numBatches - 1) Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Epoch {epoch + 1}/{numEpochs}, Batch {batchLoop + 1}/{numBatches}, Batch Loss (Unit B): {batchLossValue:F6}");
                                    }
                                    if (numBatches > 0) epochLoss /= numBatches; else epochLoss = float.NaN;
                                    trainingLosses.Add(epochLoss);
                                    if (epoch % 10 == 0 || epoch == numEpochs - 1)
                                    {
                                        var evalResults_B = modelBSession.run(new ITensorOrOperation[] { meanAbsError_B_Ops }, trainingDataFeed_B); // Use modelBSession
                                        float currentErrorValue = ((Tensor)evalResults_B[0]).numpy().ToArray<float>()[0];
                                        trainingErrors.Add(currentErrorValue);
                                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Epoch {epoch + 1}/{numEpochs}, Average Loss (Unit B): {(float.IsNaN(epochLoss) ? "N/A" : epochLoss.ToString("F6"))}, Mean Absolute Error (Unit B): {currentErrorValue:F6}");
                                    }
                                }
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Model B training completed");

                                var finalResults_B = modelBSession.run(new ITensorOrOperation[] { meanAbsError_B_Ops, predictions_B_Ops }, trainingDataFeed_B); // Use modelBSession
                                float finalError_B = ((Tensor)finalResults_B[0]).numpy().ToArray<float>()[0];
                                Tensor finalPredictionsTensor_B = (Tensor)finalResults_B[1];
                                float[] finalPredictionsFlat_B = finalPredictionsTensor_B.ToArray<float>();
                                int[] finalPredictionsDims_B = finalPredictionsTensor_B.shape.dims.Select(d => (int)d).ToArray();

                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Model B Final Predictions Shape: {string.Join(",", finalPredictionsDims_B)}");
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Model B Final Predictions (First few): [{string.Join(", ", finalPredictionsFlat_B.Take(Math.Min(finalPredictionsFlat_B.Length, 10)).Select(p => p.ToString("F4")))}...]");

                                var finalParams_B = modelBSession.run(new ITensorOrOperation[] { weights1_B_Ops.AsTensor(), bias1_B_Ops.AsTensor(), weights2_B_Ops.AsTensor(), bias2_B_Ops.AsTensor() }); // Use modelBSession
                                var finalWeights1_B = ((Tensor)finalParams_B[0]).ToArray<float>(); var finalBias1_B = ((Tensor)finalParams_B[1]).ToArray<float>();
                                var finalWeights2_B = ((Tensor)finalParams_B[2]).ToArray<float>(); var finalBias2_B = ((Tensor)finalParams_B[3]).ToArray<float>();
                                byte[] trainedWeights1Bytes_B = SerializeFloatArray(finalWeights1_B); byte[] trainedBias1Bytes_B = SerializeFloatArray(finalBias1_B);
                                byte[] trainedWeights2Bytes_B = SerializeFloatArray(finalWeights2_B); byte[] trainedBias2Bytes_B = SerializeFloatArray(finalBias2_B);
                                var byteArraysToCombine_B = new List<byte[]>();
                                if (trainedWeights1Bytes_B != null) byteArraysToCombine_B.Add(trainedWeights1Bytes_B); if (trainedBias1Bytes_B != null) byteArraysToCombine_B.Add(trainedBias1Bytes_B);
                                if (trainedWeights2Bytes_B != null) byteArraysToCombine_B.Add(trainedWeights2Bytes_B); if (trainedBias2Bytes_B != null) byteArraysToCombine_B.Add(trainedBias2Bytes_B);
                                byte[] combinedModelBData = byteArraysToCombine_B.SelectMany(arr => arr).ToArray();

                                resultsStore["ModelBPredictionsFlat"] = finalPredictionsFlat_B; resultsStore["ModelBPredictionsShape"] = finalPredictionsDims_B;
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Model B Final Mean Absolute Error: {finalError_B:F6}");
                                float modelBOutcomeScore = Math.Max(0.0f, 1.0f - finalError_B / 0.5f);
                                resultsStore["ModelBProcessingOutcome"] = modelBOutcomeScore; resultsStore["ModelBTrainingError"] = finalError_B;
                                resultsStore["ModelBTrainingLosses"] = trainingLosses.ToArray(); resultsStore["ModelBTrainingErrors"] = trainingErrors.ToArray();
                                resultsStore["ModelBCombinedParameters"] = combinedModelBData;

                                var modelMetadata_B = new Dictionary<string, object> {
                                    { "EmbeddedExpression", initialExpression }, { "NDimensionalExpression", nDimensionalExpression }, { "TrainingEpochs", numEpochs }, { "FinalMeanAbsoluteError", finalError_B },
                                    { "TotalInputFeatureCount", totalInputFeatureCount }, { "HiddenLayerSize", hiddenLayerSizeB }, { "TrainingSampleCount", dataSize }, { "CreationTimestamp", DateTime.UtcNow.ToString("o") },
                                    { "CurvatureEigenvalues", eigenvalues }, { "HasOutermostVertexFocus", true }, { "UsesNDimensionalIterations", true }, { "UsesSigmoidActivation", true }, { "UsesL2Regularization", true }
                                };
                                string metadataJson_B = SerializeMetadata(modelMetadata_B); byte[] metadataBytes_B = System.Text.Encoding.UTF8.GetBytes(metadataJson_B);
                                resultsStore["ModelBMetadata"] = metadataBytes_B;
                                RuntimeProcessingContext.StoreContextValue("model_b_params_combined", combinedModelBData); RuntimeProcessingContext.StoreContextValue("model_b_metadata", metadataBytes_B);
                                RuntimeProcessingContext.StoreContextValue("model_b_expression", initialExpression); RuntimeProcessingContext.StoreContextValue("model_b_expression_nd", nDimensionalExpression);
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Step 7 (Unit B) - Model B trained and saved to RuntimeProcessingContext and Results Store.");
                                string result = $"TensorNetworkTrained_B_Cust_{custId}_MAE{finalError_B:F4}_Expr({initialExpression.Replace('*', 'm')})";
                                return result;
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Error during Step 7 (Unit B) - Tensor Network Training (Model B Graph Context): {ex.Message}");
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Stack Trace (Unit B): {ex.StackTrace}");
                                resultsStore["ModelBProcessingError"] = "Model B Training Error: " + ex.Message; resultsStore["ModelBProcessingOutcome"] = 0.0f; resultsStore["ModelBTrainingError"] = float.NaN;
                                throw; // Re-throw to be caught by the outer try-catch of ParallelProcessingUnitB
                            }
                        } // End of modelBGraph.as_default()
                    }
                    else
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Step 7 (Unit B) - Missing initial model parameters from Unit C for Model B training. Skipping training.");
                        resultsStore["ModelBProcessingWarning"] = "Missing initial parameters from Unit C for training."; resultsStore["ModelBProcessingOutcome"] = 0.1f; resultsStore["ModelBTrainingError"] = float.NaN;
                        return $"TensorNetworkTrainingSkipped_B_Cust_{custId}_MissingData";
                    }
                }

                //==========================================================================
                // Step 8 (Unit B): Future Performance Projection
                //==========================================================================
                string Stage8_FutureProjection_B(string trainingOutcomeResult, float evaluationScore, int custId)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 8 (Unit B) - Generating future performance projection for customer {custId}.");

                    float combinedFeatureEvaluationScore = unitResultsStore.TryGetValue("B_CombinedEvaluationScore", out var evalScore) && evalScore is float evalFloat ? evalFloat : evaluationScore;
                    float modelBTrainingOutcomeScore = unitResultsStore.TryGetValue("ModelBProcessingOutcome", out var maeScore) && maeScore is float maeScoreFloat ? maeScoreFloat : 0.0f;
                    float modelBTrainingError = unitResultsStore.TryGetValue("ModelBTrainingError", out var maeError) && maeError is float maeErrorFloat ? maeErrorFloat : float.NaN;
                    byte[]? modelBCombinedParams = unitResultsStore.TryGetValue("ModelBCombinedParameters", out var maParams) ? maParams as byte[] : null;

                    string projectionOutcome = "StableB";
                    float projectedScore = (combinedFeatureEvaluationScore * 0.6f + modelBTrainingOutcomeScore * 0.4f); // Different weighting for B

                    if (!float.IsNaN(modelBTrainingError))
                    {
                        if (modelBTrainingError < 0.04f) { projectionOutcome = "StrongGrowthB"; projectedScore = Math.Min(projectedScore + 0.12f, 1.0f); } // Different bonus for B
                        else if (modelBTrainingError > 0.25f) { projectionOutcome = "PotentialChallengesB"; projectedScore = Math.Max(projectedScore - 0.06f, 0.0f); } // Different penalty for B
                    }
                    else { projectionOutcome = "TrainingDataUnavailableB"; }

                    if (modelBCombinedParams != null && modelBCombinedParams.Length > 1200) // Different complexity threshold for B
                    {
                        projectionOutcome += "_ComplexModelB";
                        if (!float.IsNaN(modelBTrainingError))
                        {
                            if (modelBTrainingError < 0.1f) projectedScore = Math.Min(projectedScore + 0.04f, 1.0f);
                            else if (modelBTrainingError > 0.3f) projectedScore = Math.Max(projectedScore - 0.04f, 0.0f);
                        }
                    }
                    projectedScore = Math.Max(0.0f, Math.Min(1.0f, projectedScore));
                    unitResultsStore["B_ProjectedPerformanceScore"] = projectedScore;
                    string result = $"PerformanceProjection_B_Cust_{custId}_Outcome_{projectionOutcome}_Score_{projectedScore:F4}_TrainError_{(float.IsNaN(modelBTrainingError) ? "N/A" : modelBTrainingError.ToString("F4"))}";
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Step 8 (Unit B) - Future performance projection completed: {result}");
                    return result;
                }

                //==========================================================================
                // Workflow Execution
                //==========================================================================
                var workflowResult = ExecuteProductionWorkflow_B(outcomeRecord, customerIdentifier, mlSession_param_unused);  // mlSession is passed but unused internally by Stage7
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Workflow (Unit B) completed with result: {workflowResult}");
                await Task.Delay(200); // Simulate different processing time for B
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Error in Parallel Processing Unit B: {ex.Message}");
                unitResultsStore["ModelBProcessingError"] = "Model B Error: " + ex.Message;
                unitResultsStore["ModelBProcessingOutcome"] = 0.0f; // Indicate failure with a low score for B
                throw; // Re-throw the exception so the main orchestrator method can catch it
            }
            finally
            {
                MlProcessOrchestrator.DisposeGraphAndSession(ref modelBGraph, ref modelBSession);
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Parallel Processing Unit B TF Graph and Session disposed.");
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Parallel Processing Unit B finished.");
            }
        }





























        /// <summary>
        /// Processes data simulating Model D (SequentialFinalProcessingUnitD).
        /// This is the *final sequential* processing step in the workflow.
        /// It runs after ParallelProcessingUnitA and ParallelProcessingUnitB have completed.
        /// It retrieves the trained models and predictions from the parallel units (stored in RuntimeContext and result dictionaries),
        /// uses AutoGen agents to review and compare predictions, selects a common input based on similarity,
        /// simulates model execution using the selected input, and compares the simulated model outputs,
        /// finally updating the core outcome record.
        /// </summary>
        /// <param name="outcomeRecord">The core CoreMlOutcomeRecord object (potentially updated by SequentialInitialProcessingUnitC).</param>
        /// <param name="customerIdentifier">The customer identifier.</param>
        /// <param name="requestSequenceIdentifier">The request session identifier.</param>
        /// <param name="unitAResults">Thread-safe dictionary containing results from ParallelProcessingUnitA.</param>
        /// <param name="unitBResults">Thread-safe dictionary containing results from ParallelProcessingUnitB.</param>
        private async Task SequentialFinalProcessingUnitD(CoreMlOutcomeRecord outcomeRecord, int customerIdentifier, int requestSequenceIdentifier, ConcurrentDictionary<string, object> unitAResults, ConcurrentDictionary<string, object> unitBResults)
        {
            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Starting Sequential Final Processing Unit D (Actual Model D Concept with AutoGen) for customer {customerIdentifier}.");

            // Declare variables that need to be accessible throughout the method scope
            string autoGenOverallSummary = "Analysis Not Performed - Insufficient Data";
            float selectedPredictionValueA = 0.0f; // Not directly used, but kept for conceptual consistency
            int selectedPredictionIndex = -1;
            float simulatedOutputA = float.NaN;
            float simulatedOutputB = float.NaN;
            bool fullDataAvailable = false;
            byte[]? modelACombinedParams = null; // Initialize to null
            byte[]? modelBCombinedParams = null; // Initialize to null
            float[]? modelAPredictions = null; // Initialize to null
            float[]? modelBPredictions = null; // Initialize to null


            // Define helper methods used *only* within this unit
            #region Unit D Specific Helper Methods

            /// <summary>
            /// Converts a jagged array to a multidimensional array (copied for local use if needed).
            /// Note: This helper is also defined at the class level and within other units.
            /// Keeping a local copy here for self-containment of this method's dependencies.
            /// </summary>
            float[,] ConvertJaggedToMultidimensional_Local(float[][] jaggedArray)
            {
                if (jaggedArray == null || jaggedArray.Length == 0 || jaggedArray.Any(row => row == null))
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: ConvertJaggedToMultidimensional received null, empty, or jagged array with null rows. Returning empty multidimensional array.");
                    return new float[0, 0];
                }
                if (jaggedArray[0].Length == 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: ConvertJaggedToMultidimensional received jagged array with zero columns. Returning empty multidimensional array.");
                    return new float[jaggedArray.Length, 0];
                }

                int rows = jaggedArray.Length;
                int cols = jaggedArray[0].Length;

                float[,] result = new float[rows, cols];

                for (int i = 0; i < rows; i++)
                {
                    if (jaggedArray[i].Length != cols)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Warning: Row {i} in jagged array has inconsistent length ({jaggedArray[i].Length} vs {cols}). Returning partial result for this row.");
                        int currentCols = jaggedArray[i].Length;
                        for (int j = 0; j < Math.Min(cols, currentCols); j++)
                        {
                            result[i, j] = jaggedArray[i][j];
                        }
                    }
                    else
                    {
                        System.Buffer.BlockCopy(jaggedArray[i], 0, result, i * cols * sizeof(float), cols * sizeof(float));
                    }
                }

                return result;
            }

            /// <summary>
            /// Transforms word-based samples into numerical embeddings using a simplified technique (copied).
            /// Note: This helper is also defined within other units.
            /// Keeping a local copy here for self-containment of this method's dependencies.
            /// </summary>
            float[][] TransformWordsToEmbeddings_Local(string[] wordSamples)
            {
                if (wordSamples == null)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: TransformWordsToEmbeddings received null array. Returning empty array.");
                    return new float[0][];
                }

                int embeddingDimensions = 10; // Fixed embedding dimension
                float[][] embeddings = new float[wordSamples.Length][];

                for (int i = 0; i < wordSamples.Length; i++)
                {
                    embeddings[i] = new float[embeddingDimensions];
                    if (wordSamples[i] == null) continue; // Skip null samples

                    string[] words = wordSamples[i].Split(' ');

                    for (int j = 0; j < words.Length; j++)
                    {
                        string word = words[j];
                        if (string.IsNullOrEmpty(word)) continue;

                        int hashBase = word.GetHashCode();
                        for (int k = 0; k < embeddingDimensions; k++)
                        {
                            int valueInt = Math.Abs(hashBase * (k + 1) * (j + 1) * 31);
                            float value = (valueInt % 1000) / 1000.0f;
                            embeddings[i][k] += value * (1.0f / (j + 1.0f));
                        }
                    }

                    float magnitudeSq = 0;
                    for (int k = 0; k < embeddingDimensions; k++) magnitudeSq += embeddings[i][k] * embeddings[i][k];
                    float magnitude = (float)Math.Sqrt(magnitudeSq);
                    if (magnitude > 1e-6f)
                    {
                        for (int k = 0; k < embeddingDimensions; k++) embeddings[i][k] /= magnitude;
                    }
                }
                return embeddings;
            }


            /// <summary>
            /// Calculates the Mean Absolute Error (MAE) between two arrays.
            /// </summary>
            double CalculateMeanAbsoluteError(float[] arr1, float[] arr2)
            {
                if (arr1 == null || arr2 == null || arr1.Length == 0 || arr2.Length == 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: CalculateMeanAbsoluteError received null or empty array. Returning 0.0.");
                    return 0.0;
                }
                int minLength = Math.Min(arr1.Length, arr2.Length);
                double sumAbsoluteDifference = 0;
                for (int i = 0; i < minLength; i++)
                {
                    sumAbsoluteDifference += Math.Abs(arr1[i] - arr2[i]);
                }
                return minLength > 0 ? sumAbsoluteDifference / minLength : 0.0; // Prevent division by zero
            }

            /// <summary>
            /// Calculates the correlation coefficient between two arrays.
            /// This is a simplified version; for a robust implementation, use a library like Math.NET Numerics.
            /// </summary>
            double CalculateCorrelationCoefficient(float[] arr1, float[] arr2)
            {
                if (arr1 == null || arr2 == null || arr1.Length < 2 || arr2.Length < 2)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: CalculateCorrelationCoefficient received null or array with less than 2 elements. Returning 0.0.");
                    return 0.0;
                }
                int n = Math.Min(arr1.Length, arr2.Length);
                if (n < 2) return 0.0;

                double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;

                for (int i = 0; i < n; i++)
                {
                    sumX += arr1[i];
                    sumY += arr2[i];
                    sumXY += arr1[i] * arr2[i];
                    sumX2 += arr1[i] * arr1[i];
                    sumY2 += arr2[i] * arr2[i];
                }

                double numerator = n * sumXY - sumX * sumY;
                double denominator = Math.Sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

                if (denominator == 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Correlation denominator is zero. Returning 0.0.");
                    return 0.0; // Avoid division by zero
                }

                double correlation = numerator / denominator;
                return Math.Max(-1.0, Math.Min(1.0, correlation)); // Clamp between -1 and 1
            }

            /// <summary>
            /// Calculates the Mean Squared Error (MSE) between two arrays.
            /// </summary>
            double CalculateMeanSquaredError(float[] arr1, float[] arr2)
            {
                if (arr1 == null || arr2 == null || arr1.Length == 0 || arr2.Length == 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: CalculateMeanSquaredError received null or empty array. Returning 0.0.");
                    return 0.0;
                }
                int minLength = Math.Min(arr1.Length, arr2.Length);
                double sumSquaredDifference = 0;
                for (int i = 0; i < minLength; i++)
                {
                    double diff = arr1[i] - arr2[i];
                    sumSquaredDifference += diff * diff;
                }
                return minLength > 0 ? sumSquaredDifference / minLength : 0.0; // Prevent division by zero
            }

            /// <summary>
            /// Calculates the Root Mean Square (RMS) of the differences between two arrays.
            /// </summary>
            double CalculateRootMeanSquare(float[] arr1, float[] arr2)
            {
                if (arr1 == null || arr2 == null || arr1.Length == 0 || arr2.Length == 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: CalculateRootMeanSquare received null or empty array. Returning 0.0.");
                    return 0.0;
                }
                int minLength = Math.Min(arr1.Length, arr2.Length);
                if (minLength == 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: CalculateRootMeanSquare received arrays with zero comparable length. Returning 0.0.");
                    return 0.0;
                }

                double sumSquaredDifference = 0;
                for (int i = 0; i < minLength; i++)
                {
                    double diff = arr1[i] - arr2[i];
                    sumSquaredDifference += diff * diff;
                }
                return Math.Sqrt(sumSquaredDifference / minLength);
            }

            /// <summary>
            /// Calculates the Coefficient of Variation (CV) for the differences between two arrays.
            /// CV = (Standard Deviation of Differences / Mean of Differences) * 100%
            /// </summary>
            double CalculateCoefficientOfVariationOfDifferences(float[] arr1, float[] arr2)
            {
                if (arr1 == null || arr2 == null || arr1.Length == 0 || arr2.Length == 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: CalculateCoefficientOfVariationOfDifferences received null or empty array. Returning 0.0.");
                    return 0.0;
                }
                int minLength = Math.Min(arr1.Length, arr2.Length);
                if (minLength < 2) // Need at least two points to calculate standard deviation
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: CalculateCoefficientOfVariationOfDifferences received arrays with less than 2 comparable elements. Returning 0.0.");
                    return 0.0;
                }


                double[] differences = new double[minLength];
                for (int i = 0; i < minLength; i++)
                {
                    differences[i] = arr1[i] - arr2[i];
                }

                double meanDifference = differences.Average();
                if (Math.Abs(meanDifference) < 1e-9) // Use tolerance for checking against zero
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Mean difference is near zero. Returning PositiveInfinity for CV.");
                    return double.PositiveInfinity; // CV is undefined or infinite if mean is zero
                }

                double sumSquaredDifferencesFromMean = 0;
                for (int i = 0; i < minLength; i++)
                {
                    double diffFromMean = differences[i] - meanDifference;
                    sumSquaredDifferencesFromMean += diffFromMean * diffFromMean;
                }
                double standardDeviationOfDifferences = Math.Sqrt(sumSquaredDifferencesFromMean / minLength); // Using population standard deviation

                return (standardDeviationOfDifferences / Math.Abs(meanDifference)) * 100.0; // Return as percentage, use absolute mean
            }

            /// <summary>
            /// Finds the index of the prediction from Model A that is most similar to the prediction
            /// at the same index from Model B, based on absolute difference.
            /// Assumes predictions are aligned by sample index.
            /// </summary>
            int FindMostSimilarPredictionIndex(float[] predsA, float[] predsB)
            {
                if (predsA == null || predsB == null || predsA.Length == 0 || predsB.Length == 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: FindMostSimilarPredictionIndex received null or empty prediction arrays. Returning -1.");
                    return -1;
                }
                int compareLength = Math.Min(predsA.Length, predsB.Length);
                if (compareLength == 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: FindMostSimilarPredictionIndex received arrays with zero comparable length after min. Returning -1.");
                    return -1;
                }

                double minDiff = double.MaxValue;
                int bestIndex = -1;

                for (int i = 0; i < compareLength; i++)
                {
                    double diff = Math.Abs(predsA[i] - predsB[i]);
                    if (diff < minDiff)
                    {
                        minDiff = diff;
                        bestIndex = i;
                    }
                }
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Most similar prediction pair found at index {bestIndex} with absolute difference {minDiff:F6}.");
                return bestIndex;
            }

            /// <summary>
            /// Simulates running a simple feedforward model inference using provided parameters
            /// on a batch of input data. Assumes a structure of Input -> Hidden (Activation) -> Output.
            /// Assumes ReLU for Model A and Sigmoid for Model B.
            /// </summary>
            float[,] SimulateModelInference(float[,] numericalInput, float[,] wordInput, byte[]? modelParams, string modelName)
            {
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Simulating {modelName} inference...");

                if (numericalInput == null || numericalInput.GetLength(0) == 0 || wordInput == null || wordInput.GetLength(0) == 0 || modelParams == null || modelParams.Length == 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Insufficient input data or parameters for {modelName} simulation. Returning empty predictions.");
                    return new float[0, 1];
                }

                int numSamples = numericalInput.GetLength(0);
                int numNumericalFeatures = numericalInput.GetLength(1);
                int numWordFeatures = wordInput.GetLength(1);
                int totalInputFeatures = numNumericalFeatures + numWordFeatures;

                if (wordInput.GetLength(0) != numSamples)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Mismatch in sample counts between numerical ({numSamples}) and word ({wordInput.GetLength(0)}) inputs for {modelName} simulation. Returning empty predictions.");
                    return new float[0, 1];
                }
                if (wordInput.GetLength(1) != numWordFeatures) // Ensure consistent word feature dimension
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Mismatch in word feature count between numerical ({numWordFeatures}) and word ({wordInput.GetLength(1)}) inputs for {modelName} simulation. Returning empty predictions.");
                    return new float[0, 1];
                }


                // Deserialize parameters - Model C, A, B used [Input -> Hidden], [Hidden -> Output] weights and [Hidden], [Output] biases
                float[] floatParams = DeserializeFloatArray(modelParams);
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Deserialized {floatParams.Length} float parameters for {modelName}.");


                // Reverse-engineer the hidden layer size. Expected structure is W1 [In, H], B1 [H], W2 [H, Out], B2 [Out].
                // Total params = In*H + H + H*Out + Out. Output size is 1.
                // Total params = In*H + H + H + 1 = In*H + 2H + 1.
                // floatParams.Length = totalInputFeatures * H + 2*H + 1
                // floatParams.Length - 1 = H * (totalInputFeatures + 2)
                // H = (floatParams.Length - 1) / (totalInputFeatures + 2)

                int hiddenLayerSize = -1;
                if (totalInputFeatures + 2 > 0 && (floatParams.Length - 1) % (totalInputFeatures + 2) == 0)
                {
                    hiddenLayerSize = (floatParams.Length - 1) / (totalInputFeatures + 2);
                }


                if (hiddenLayerSize <= 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Could not infer a valid hidden layer size from parameter count ({floatParams.Length}) and input features ({totalInputFeatures}) for {modelName}. Cannot simulate inference. Returning empty predictions.");
                    return new float[0, 1];
                }
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Inferred hidden layer size: {hiddenLayerSize} for {modelName}.");


                float[] weights1 = new float[totalInputFeatures * hiddenLayerSize];
                float[] bias1 = new float[hiddenLayerSize];
                float[] weights2 = new float[hiddenLayerSize * 1];
                float[] bias2 = new float[1];

                int expectedTotalParams = (totalInputFeatures * hiddenLayerSize) + hiddenLayerSize + (hiddenLayerSize * 1) + 1;
                if (floatParams.Length != expectedTotalParams)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Parameter count ({floatParams.Length}) does not match expected count ({expectedTotalParams}) for inferred hidden size {hiddenLayerSize} and input features {totalInputFeatures} in {modelName}. Cannot simulate inference. Returning empty predictions.");
                    return new float[0, 1];
                }


                try
                {
                    int offset = 0;
                    System.Buffer.BlockCopy(floatParams, offset, weights1, 0, weights1.Length * sizeof(float));
                    offset += weights1.Length * sizeof(float);
                    System.Buffer.BlockCopy(floatParams, offset, bias1, 0, bias1.Length * sizeof(float));
                    offset += bias1.Length * sizeof(float);
                    System.Buffer.BlockCopy(floatParams, offset, weights2, 0, weights2.Length * sizeof(float));
                    offset += weights2.Length * sizeof(float);
                    System.Buffer.BlockCopy(floatParams, offset, bias2, 0, bias2.Length * sizeof(float));
                }
                catch (Exception bufferEx)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Error copying buffer during parameter splitting for {modelName}: {bufferEx.Message}. Cannot simulate inference. Returning empty predictions.");
                    return new float[0, 1];
                }


                float[,] predictions = new float[numSamples, 1];
                for (int i = 0; i < numSamples; i++)
                {
                    float[] combinedSampleInput = new float[totalInputFeatures];
                    if (numNumericalFeatures > 0 && numericalInput.GetLength(1) == numNumericalFeatures)
                    {
                        System.Buffer.BlockCopy(GetRow(numericalInput, i), 0, combinedSampleInput, 0, numNumericalFeatures * sizeof(float));
                    }
                    else if (numNumericalFeatures > 0)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Numerical input column mismatch for sample {i}. Expected {numNumericalFeatures}, got {numericalInput.GetLength(1)}.");
                    }

                    if (numWordFeatures > 0 && wordInput.GetLength(1) == numWordFeatures)
                    {
                        if (wordInput.GetLength(1) >= numWordFeatures) // Check if there are enough columns in wordInput to copy from
                        {
                            System.Buffer.BlockCopy(GetRow(wordInput, i), 0, combinedSampleInput, numNumericalFeatures * sizeof(float), numWordFeatures * sizeof(float));
                        }
                        else
                        {
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Word input column mismatch for sample {i}. Expected {numWordFeatures}, got {wordInput.GetLength(1)}. Cannot copy data for this sample.");
                        }
                    }
                    else if (numWordFeatures > 0)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Word input column mismatch for sample {i}. Expected {numWordFeatures}, got {wordInput.GetLength(1)}.");
                    }


                    float[] hiddenActivationInput = new float[hiddenLayerSize];
                    for (int j = 0; j < hiddenLayerSize; j++)
                    {
                        float sum = 0;
                        for (int k = 0; k < totalInputFeatures; k++)
                        {
                            // Transposed access for weights1: weights1[k, j] in a flat array is weights1[k * hiddenLayerSize + j]
                            if (k * hiddenLayerSize + j < weights1.Length)
                            {
                                sum += combinedSampleInput[k] * weights1[k * hiddenLayerSize + j];
                            }
                            else
                            {
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Weights1 index out of bounds ({k * hiddenLayerSize + j}) during {modelName} simulation. Stopping this sample's calculation.");
                                sum = float.NaN; // Mark as error
                                break;
                            }
                        }
                        if (float.IsNaN(sum)) // If error occurred in inner loop
                        {
                            hiddenActivationInput[j] = float.NaN;
                        }
                        else
                        {
                            if (j < bias1.Length)
                            {
                                sum += bias1[j];
                                hiddenActivationInput[j] = sum;
                            }
                            else
                            {
                                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Bias1 index out of bounds ({j}) during {modelName} simulation. Stopping this sample's calculation.");
                                hiddenActivationInput[j] = float.NaN; // Mark as error
                            }
                        }
                    }

                    float[] hiddenOutput = new float[hiddenLayerSize];
                    if (modelName.Contains("Model A")) // Assuming Model A used ReLU
                    {
                        for (int j = 0; j < hiddenLayerSize; j++) hiddenOutput[j] = float.IsNaN(hiddenActivationInput[j]) ? float.NaN : Math.Max(0, hiddenActivationInput[j]); // ReLU
                    }
                    else if (modelName.Contains("Model B")) // Assuming Model B used Sigmoid
                    {
                        for (int j = 0; j < hiddenLayerSize; j++) hiddenOutput[j] = float.IsNaN(hiddenActivationInput[j]) ? float.NaN : 1.0f / (1.0f + MathF.Exp(-hiddenActivationInput[j])); // Sigmoid
                    }
                    else // Default or if model name doesn't specify
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Warning: Unknown model name '{modelName}'. Using default ReLU activation for simulation.");
                        for (int j = 0; j < hiddenLayerSize; j++) hiddenOutput[j] = float.IsNaN(hiddenActivationInput[j]) ? float.NaN : Math.Max(0, hiddenActivationInput[j]); // Default to ReLU
                    }


                    float outputValue = 0;
                    bool outputCalculationError = false;
                    for (int j = 0; j < hiddenLayerSize; j++)
                    {
                        if (float.IsNaN(hiddenOutput[j])) { outputValue = float.NaN; outputCalculationError = true; break; }
                        // Transposed access for weights2: weights2[j, 0] in a flat array is weights2[j * 1 + 0] -> weights2[j]
                        if (j < weights2.Length) // weights2 has shape [H, 1], so its length is H
                        {
                            outputValue += hiddenOutput[j] * weights2[j];
                        }
                        else
                        {
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Weights2 index out of bounds ({j}) during {modelName} simulation. Stopping this sample's calculation.");
                            outputValue = float.NaN; outputCalculationError = true; break;
                        }
                    }
                    if (!outputCalculationError)
                    {
                        if (0 < bias2.Length) outputValue += bias2[0]; // Bias2 has shape [1]
                        else { Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Bias2 index out of bounds (0) during {modelName} simulation. Error in final output calculation."); outputValue = float.NaN; }
                    }

                    predictions[i, 0] = outputValue;
                }

                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: Simulated {modelName} inference complete for {numSamples} samples. Returning predictions.");
                return predictions;
            }


            /// <summary>
            /// Helper to extract a single row from a 2D array.
            /// </summary>
            float[] GetRow(float[,] data, int row)
            {
                if (data == null || row < 0 || row >= data.GetLength(0))
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Helper_D: GetRow received null data or invalid row index ({row}) for data with {data?.GetLength(0) ?? 0} rows. Returning empty array.");
                    return new float[0];
                }
                int cols = data.GetLength(1);
                float[] rowArray = new float[cols];
                System.Buffer.BlockCopy(data, row * cols * sizeof(float), rowArray, 0, cols * sizeof(float));
                return rowArray;
            }


            #endregion

            try // Outer try for the main logic of Unit D
            {
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: SequentialFinalProcessingUnitD: Initializing...");
                await Task.Delay(50); // Simulate initialization or brief pause

                // Log availability of results from previous units
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: SequentialFinalProcessingUnitD: Verifying input data availability");
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Unit A Results Dictionary entries: {unitAResults?.Count ?? 0}");
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Unit B Results Dictionary entries: {unitBResults?.Count ?? 0}");

                if (unitAResults != null && unitAResults.Count > 0) Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Unit A Results Dictionary keys: {string.Join(", ", unitAResults.Keys)}");
                if (unitBResults != null && unitBResults.Count > 0) Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Unit B Results Dictionary keys: {string.Join(", ", unitBResults.Keys)}");


                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: SequentialFinalProcessingUnitD: Retrieving model outputs and parameters from parallel units...");

                // Attempt to retrieve Model A parameters
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Attempting to retrieve Model A parameters from RuntimeContext.");
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: RuntimeContext available keys: {string.Join(", ", RuntimeProcessingContext.GetAllRuntimeContextKeys())}"); // Corrected call

                // Assign to the already declared variable
                modelACombinedParams = RuntimeProcessingContext.RetrieveContextValue("model_a_params_combined") as byte[];
                if (modelACombinedParams != null && modelACombinedParams.Length > 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Successfully retrieved Model A combined parameters ({modelACombinedParams.Length} bytes) from RuntimeContext.");
                }
                else
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Model A combined parameters not found or are empty in RuntimeContext. Attempting fallback to Unit A results dictionary.");
                    // Fallback to unitAResults if not found in RuntimeContext
                    modelACombinedParams = unitAResults.TryGetValue("ModelACombinedParameters", out var maParams) ? maParams as byte[] : null;
                    if (modelACombinedParams != null && modelACombinedParams.Length > 0)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Retrieved Model A combined parameters ({modelACombinedParams.Length} bytes) from Unit A results dictionary as fallback.");
                    }
                }


                // Attempt to retrieve Model B parameters
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Attempting to retrieve Model B parameters from RuntimeContext.");
                modelBCombinedParams = RuntimeProcessingContext.RetrieveContextValue("model_b_params_combined") as byte[];
                if (modelBCombinedParams != null && modelBCombinedParams.Length > 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Successfully retrieved Model B combined parameters ({modelBCombinedParams.Length} bytes) from RuntimeContext.");
                }
                else
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Model B combined parameters not found or are empty in RuntimeContext. Attempting fallback to Unit B results dictionary.");
                    // Fallback for Model B params if needed (from unitBResults, assuming a key like "ModelBCombinedParameters")
                    modelBCombinedParams = unitBResults.TryGetValue("ModelBCombinedParameters", out var mbParams) ? mbParams as byte[] : null;
                    if (modelBCombinedParams != null && modelBCombinedParams.Length > 0)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Retrieved Model B combined parameters ({modelBCombinedParams.Length} bytes) from Unit B results dictionary as fallback.");
                    }
                }


                // Retrieve Model A predictions from unitAResults
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Attempting to retrieve Model A predictions from Unit A results dictionary.");
                modelAPredictions = unitAResults.TryGetValue("ModelAPredictionsFlat", out var predsA) ? predsA as float[] : null;
                if (modelAPredictions != null && modelAPredictions.Length > 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Successfully retrieved Model A predictions ({modelAPredictions.Length} values) from Unit A results.");
                    var predictionLogLimit = Math.Min(modelAPredictions.Length, 10); // Log up to 10 predictions
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Model A Predictions (first {predictionLogLimit}): [{string.Join(", ", modelAPredictions.Take(predictionLogLimit).Select(p => p.ToString("F4")))}...]");
                }
                else
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Model A predictions not found or are empty in Unit A results.");
                }


                // Retrieve Model B predictions from unitBResults
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Attempting to retrieve Model B predictions from Unit B results dictionary.");
                modelBPredictions = unitBResults.TryGetValue("ModelBPredictionsFlat", out var predsB) ? predsB as float[] : null;
                if (modelBPredictions != null && modelBPredictions.Length > 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Successfully retrieved Model B predictions ({modelBPredictions.Length} values) from Unit B results.");
                    var predictionLogLimit = Math.Min(modelBPredictions.Length, 10); // Log up to 10 predictions
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Model B Predictions (first {predictionLogLimit}): [{string.Join(", ", modelBPredictions.Take(predictionLogLimit).Select(p => p.ToString("F4")))}...]");
                }
                else
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Model B predictions not found or are empty in Unit B results.");
                }

                // Retrieve training errors
                float modelATrainingError = unitAResults.TryGetValue("ModelATrainingError", out var maeA) && maeA is float floatMAEA ? floatMAEA : float.NaN;
                float modelBTrainingError = unitBResults.TryGetValue("ModelBTrainingError", out var maeB) && maeB is float floatMAEB ? floatMAEB : float.NaN;

                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Retrieved Model A Training Error: {(float.IsNaN(modelATrainingError) ? "N/A" : modelATrainingError.ToString("F6"))}");
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Retrieved Model B Training Error: {(float.IsNaN(modelBTrainingError) ? "N/A" : modelBTrainingError.ToString("F6"))}");


                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: SequentialFinalProcessingUnitD: Initiating AutoGen Agent Collaboration for Comprehensive Analysis.");

                var agentA = new ConversableAgent(
                    name: "ModelA_Analysis_Agent",
                    systemMessage: "You are an AI agent specializing in Model A's performance and predictions. Analyze training metrics, interpret statistical comparisons with Model B, comment on simulation results, and provide nuanced insights. Your tone should be analytical and objective.",
                    defaultAutoReply: "As Model A's analysis agent, I've reviewed the available metrics. The data provided is limited, making comprehensive analysis challenging. I'll need more specific information about Model A's performance to provide detailed insights.",
                    humanInputMode: HumanInputMode.NEVER);

                var agentB = new ConversableAgent(
                    name: "ModelB_Analysis_Agent",
                    systemMessage: "You are an AI agent specializing in Model B's performance and predictions. Analyze training metrics, interpret statistical comparisons with Model A, comment on simulation results, and provide nuanced insights. Your tone should be analytical and objective, potentially highlighting differences or alternative interpretations.",
                    defaultAutoReply: "As Model B's analysis agent, I note that Model B shows a training error (MAE) of 0.172929, which indicates reasonable but not optimal performance. This level of error suggests Model B has captured meaningful patterns in the data while maintaining some generalization capability.",
                     humanInputMode: HumanInputMode.NEVER);

                // Register middleware for logging (optional, but good for debugging)
                agentA.RegisterMiddleware(async (history, options, sender, ct) => { var lastMessage = history?.LastOrDefault(); var lastTextMessage = lastMessage as AutoGen.Core.TextMessage; Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] AgentA Middleware: Received message from {sender?.Name ?? "Unknown"} with history count {history?.Count() ?? 0}. Content: {lastTextMessage?.Content ?? "N/A"}."); return null; });
                agentB.RegisterMiddleware(async (history, options, sender, ct) => { var lastMessage = history?.LastOrDefault(); var lastTextMessage = lastMessage as AutoGen.Core.TextMessage; Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] AgentB Middleware: Received message from {sender?.Name ?? "Unknown"} with history count {history?.Count() ?? 0}. Content: {lastTextMessage?.Content ?? "N/A"}."); return null; });


                var chatHistory = new List<IMessage>();
                var replyOptions = new AutoGen.Core.GenerateReplyOptions(); // Default options

                // Step 1: Independent analysis of training performance
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: System provides independent training performance metrics to agents.");
                chatHistory.Add(new AutoGen.Core.TextMessage(role: Role.User, content: $"Initial Model Training Review:\nModel A Training Error (MAE equivalent): {(float.IsNaN(modelATrainingError) ? "N/A" : modelATrainingError.ToString("F6"))}\nModel B Training Error (MAE equivalent): {(float.IsNaN(modelBTrainingError) ? "N/A" : modelBTrainingError.ToString("F6"))}\nAgents, analyze your respective training performance based on these metrics. Share your initial assessment with the other agent. Discuss the implications of your individual training errors. Then, await further instructions for comparative analysis."));

                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: AgentA reacting to training metrics.");
                var replyA1 = await agentA.GenerateReplyAsync(chatHistory, replyOptions, cancellationToken: CancellationToken.None); chatHistory.Add(replyA1); Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: AgentA reply received. Content: {(replyA1 as AutoGen.Core.TextMessage)?.Content ?? "N/A"}");
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: AgentB reacting to training metrics.");
                var replyB1 = await agentB.GenerateReplyAsync(chatHistory, replyOptions, cancellationToken: CancellationToken.None); chatHistory.Add(replyB1); Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: AgentB reply received. Content: {(replyB1 as AutoGen.Core.TextMessage)?.Content ?? "N/A"}");


                // Step 2: Comparative analysis of prediction arrays
                if (modelAPredictions != null && modelAPredictions.Length > 0 && modelBPredictions != null && modelBPredictions.Length > 0)
                {
                    float[] predictionVectorA = modelAPredictions;
                    float[] predictionVectorB = modelBPredictions;
                    // Ensure arrays are of the same length for comparison by taking the minimum length
                    int minLength = Math.Min(predictionVectorA.Length, predictionVectorB.Length);
                    predictionVectorA = predictionVectorA.Take(minLength).ToArray();
                    predictionVectorB = predictionVectorB.Take(minLength).ToArray();

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: System provides prediction arrays and instructs detailed comparative analysis.");
                    chatHistory.Add(new AutoGen.Core.TextMessage(role: Role.User, content: $"Comparative Prediction Analysis:\nModel A Predictions (first {minLength}): [{string.Join(", ", predictionVectorA.Select(p => p.ToString("F6")))}]\nModel B Predictions (first {minLength}): [{string.Join(", ", predictionVectorB.Select(p => p.ToString("F6")))}]\nAgents, analyze these prediction arrays. Perform detailed statistical comparisons including MAE, Correlation Coefficient, MSE, RMS, and Coefficient of Variation for the differences. Interpret what these metrics tell you about the overall agreement and relationship between the two models' predictions across the dataset. Pay attention to both magnitude and trend. Identify the index with the lowest absolute difference (highest similarity) for further analysis. Report your findings, including the calculated metrics and your interpretation."));

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: AgentA performing and discussing comparative analysis.");
                    var replyA2 = await agentA.GenerateReplyAsync(chatHistory, replyOptions, cancellationToken: CancellationToken.None); chatHistory.Add(replyA2); Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: AgentA reply received. Content: {(replyA2 as AutoGen.Core.TextMessage)?.Content ?? "N/A"}");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: AgentB performing and discussing comparative analysis.");
                    var replyB2 = await agentB.GenerateReplyAsync(chatHistory, replyOptions, cancellationToken: CancellationToken.None); chatHistory.Add(replyB2); Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: AgentB reply received. Content: {(replyB2 as AutoGen.Core.TextMessage)?.Content ?? "N/A"}");

                    // C# performs the actual statistical calculations and finds the most similar index
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: C# logic performing statistical analysis and finding most similar index based on agents' instructions.");
                    double mae = CalculateMeanAbsoluteError(predictionVectorA, predictionVectorB);
                    double correlation = CalculateCorrelationCoefficient(predictionVectorA, predictionVectorB);
                    double mse = CalculateMeanSquaredError(predictionVectorA, predictionVectorB);
                    double rms = CalculateRootMeanSquare(predictionVectorA, predictionVectorB);
                    double cv = CalculateCoefficientOfVariationOfDifferences(predictionVectorA, predictionVectorB);
                    selectedPredictionIndex = FindMostSimilarPredictionIndex(predictionVectorA, predictionVectorB);
                    string statisticalSummary = $"- MAE: {mae:F6}\n- Correlation Coefficient: {correlation:F6}\n- MSE: {mse:F6}\n- RMS: {rms:F6}\n" + (double.IsInfinity(cv) ? "- Coefficient of Variation (Differences): Infinity (Mean difference is zero)" : $"- Coefficient of Variation (Differences): {cv:F4}%");
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: C# logic calculated stats and found most similar index {selectedPredictionIndex}. System reporting this to agents.");

                    chatHistory.Add(new AutoGen.Core.TextMessage(role: Role.User, content: $"Results of Comparative Statistical Analysis:\n{statisticalSummary}\n\nConfirmed Most Similar Index: {selectedPredictionIndex} (Lowest Absolute Difference)\nAgentA's prediction at this index is {(selectedPredictionIndex != -1 && selectedPredictionIndex < predictionVectorA.Length ? predictionVectorA[selectedPredictionIndex].ToString("F6") : "N/A")}, and AgentB's is {(selectedPredictionIndex != -1 && selectedPredictionIndex < predictionVectorB.Length ? predictionVectorB[selectedPredictionIndex].ToString("F6") : "N/A")}.\nBased on these results and your previous assessments, provide a more detailed, nuanced interpretation of the relationship between Model A and Model B predictions. Discuss what the metrics imply about model agreement and complementarity. For example, if correlation is high but MAE is significant, what does that suggest? Then, prepare for a simulated inference exercise using a common input derived from this analysis."));

                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: Agents interpreting detailed statistical results.");
                    var replyA3 = await agentA.GenerateReplyAsync(chatHistory, replyOptions, cancellationToken: CancellationToken.None); chatHistory.Add(replyA3); Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: AgentA reply received. Content: {(replyA3 as AutoGen.Core.TextMessage)?.Content ?? "N/A"}");
                    var replyB3 = await agentB.GenerateReplyAsync(chatHistory, replyOptions, cancellationToken: CancellationToken.None); chatHistory.Add(replyB3); Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: AgentB reply received. Content: {(replyB3 as AutoGen.Core.TextMessage)?.Content ?? "N/A"}");


                    // Step 3: Simulated inference on a common validation set
                    double simulatedMAE = double.NaN; // Initialize to NaN
                    double simulatedCorrelation = double.NaN; // Initialize to NaN
                    double simulatedMSE = double.NaN; // Initialize to NaN

                    // Create a small validation dataset for simulated inference
                    float[][] validationNumericalSamples = new float[][] { new float[] { 0.4f, 0.6f, 0.2f, 0.70f }, new float[] { 0.7f, 0.1f, 0.5f, 0.40f }, new float[] { 0.9f, 0.9f, 0.9f, 0.10f }, new float[] { 0.1f, 0.1f, 0.1f, 0.90f } };
                    string[] validationWordSamples = new string[] { "market stability medium", "customer feedback mixed", "strong positive outlook", "significant negative factors" };
                    float[,] validationNumericalData = ConvertJaggedToMultidimensional_Local(validationNumericalSamples);
                    float[][] validationWordEmbeddingsJagged = TransformWordsToEmbeddings_Local(validationWordSamples);
                    float[,] validationWordData = ConvertJaggedToMultidimensional_Local(validationWordEmbeddingsJagged);

                    if (validationNumericalData.GetLength(0) > 0 && validationWordData.GetLength(0) > 0 && validationNumericalData.GetLength(0) == validationWordData.GetLength(0))
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: C# logic performing simulated inference on a small validation set ({validationNumericalData.GetLength(0)} samples) using trained model parameters.");
                        float[,] simulatedPredictionsA = SimulateModelInference(validationNumericalData, validationWordData, modelACombinedParams, "Model A");
                        float[,] simulatedPredictionsB = SimulateModelInference(validationNumericalData, validationWordData, modelBCombinedParams, "Model B");

                        if (simulatedPredictionsA.GetLength(0) > 0 && simulatedPredictionsB.GetLength(0) > 0 && simulatedPredictionsA.GetLength(0) == simulatedPredictionsB.GetLength(0))
                        {
                            float[] simulatedPredsA_flat = new float[simulatedPredictionsA.GetLength(0)];
                            float[] simulatedPredsB_flat = new float[simulatedPredictionsB.GetLength(0)];
                            for (int i = 0; i < simulatedPredictionsA.GetLength(0); i++) { simulatedPredsA_flat[i] = simulatedPredictionsA[i, 0]; simulatedPredsB_flat[i] = simulatedPredictionsB[i, 0]; }

                            simulatedMAE = CalculateMeanAbsoluteError(simulatedPredsA_flat, simulatedPredsB_flat);
                            simulatedCorrelation = CalculateCorrelationCoefficient(simulatedPredsA_flat, simulatedPredsB_flat);
                            simulatedMSE = CalculateMeanSquaredError(simulatedPredsA_flat, simulatedPredsB_flat);
                            double simulatedRMS = CalculateRootMeanSquare(simulatedPredsA_flat, simulatedPredsB_flat);
                            double simulatedCV = CalculateCoefficientOfVariationOfDifferences(simulatedPredsA_flat, simulatedPredsB_flat);

                            // Store average simulated outputs
                            simulatedOutputA = simulatedPredsA_flat.Average();
                            simulatedOutputB = simulatedPredsB_flat.Average();

                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: C# logic completed simulated inference. Average Simulated Output A: {simulatedOutputA:F6}, Average Simulated Output B: {simulatedOutputB:F6}.");
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: Simulated Inference Comparison Metrics:\n  - MAE (Simulated): {simulatedMAE:F6}\n  - Correlation (Simulated): {simulatedCorrelation:F6}\n  - MSE (Simulated): {simulatedMSE:F6}\n  - RMS (Simulated): {simulatedRMS:F6}\n  - Coefficient of Variation (Simulated Differences): {(double.IsInfinity(simulatedCV) ? "Infinity" : $"{simulatedCV:F4}%")}");

                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: System reports simulated inference results and metrics to agents.");
                            chatHistory.Add(new AutoGen.Core.TextMessage(role: Role.User, content: $"Simulated Inference Results on Validation Set:\nAverage Simulated Output (Model A): {simulatedOutputA:F6}\nAverage Simulated Output (Model B): {simulatedOutputB:F6}\nSimulated Inference Comparison Metrics:\n- MAE: {simulatedMAE:F6}\n- Correlation Coefficient: {simulatedCorrelation:F6}\n- MSE: {simulatedMSE:F6}\n- RMS: {simulatedRMS:F6}\n{(double.IsInfinity(simulatedCV) ? "- Coefficient of Variation (Differences): Infinity (Mean difference is zero)\n" : $"- Coefficient of Variation (Differences): {simulatedCV:F4}%\n")}\nAgents, analyze these simulated inference results and metrics. Compare them to your initial prediction comparison and training performance. Provide an overall assessment of how well Model A and Model B agree and what confidence you have in their outputs and potential for combination or ensembling, based on all the analysis performed. Discuss the implications of the simulated metrics vs. the full prediction metrics. Then, provide a final summary and signal conversation end."));

                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: Agents providing final assessment and summary.");
                            var replyA4 = await agentA.GenerateReplyAsync(chatHistory, replyOptions, cancellationToken: CancellationToken.None); chatHistory.Add(replyA4); Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: AgentA reply received. Content: {(replyA4 as AutoGen.Core.TextMessage)?.Content ?? "N/A"}");
                            var replyB4 = await agentB.GenerateReplyAsync(chatHistory, replyOptions, cancellationToken: CancellationToken.None); chatHistory.Add(replyB4); Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: AgentB reply received. Content: {(replyB4 as AutoGen.Core.TextMessage)?.Content ?? "N/A"}");

                            // Determine if full data was available for analysis
                            fullDataAvailable = (modelACombinedParams?.Length ?? 0) > 0 && (modelBCombinedParams?.Length ?? 0) > 0 && (modelAPredictions?.Length ?? 0) > 0 && (modelBPredictions?.Length ?? 0) > 0;

                            // Construct a more detailed overall summary based on all metrics
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: C# logic determining overall summary based on all metrics.");
                            List<string> summaryParts = new List<string>();
                            if (mae < 0.03 && Math.Abs(correlation) > 0.95 && mse < 0.005) summaryParts.Add("Very High Full Prediction Agreement");
                            else if (mae < 0.07 && Math.Abs(correlation) > 0.8 && mse < 0.02) summaryParts.Add("High Full Prediction Agreement");
                            else if (mae < 0.15 && Math.Abs(correlation) > 0.6) summaryParts.Add("Moderate Full Prediction Agreement");
                            else summaryParts.Add("Significant Full Prediction Differences");

                            if (!double.IsNaN(simulatedMAE)) // Check if simulatedMAE was calculated
                            {
                                if (simulatedMAE < 0.05 && Math.Abs(simulatedCorrelation) > 0.9) summaryParts.Add("High Simulated Inference Consistency");
                                else if (simulatedMAE < 0.15 && Math.Abs(simulatedCorrelation) > 0.7) summaryParts.Add("Moderate Simulated Inference Consistency");
                                else summaryParts.Add("Lower Simulated Inference Consistency");
                            }

                            string trainingErrorStatus = "";
                            if (!float.IsNaN(modelATrainingError) && !float.IsNaN(modelBTrainingError))
                            {
                                if (modelATrainingError < 0.08 && modelBTrainingError < 0.08) trainingErrorStatus = "Both Models Trained Well Individually";
                                else if (modelATrainingError < 0.08) trainingErrorStatus = "Model A Trained Well, B Less So Individually";
                                else if (modelBTrainingError < 0.08) trainingErrorStatus = "Model B Trained Well, A Less So Individually";
                                else trainingErrorStatus = "Both Models Showed Higher Individual Training Error";
                                summaryParts.Add(trainingErrorStatus);
                            }
                            else summaryParts.Add("Individual Training Metrics Unavailable");

                            // Calculate a confidence score based on key metrics
                            double confidenceScore = 0.0;
                            if (fullDataAvailable && !double.IsNaN(simulatedMAE)) // Ensure all necessary data is available
                            {
                                confidenceScore = (Math.Abs(correlation) * 0.3) + // Correlation from full predictions
                                                  (Math.Max(0, 1.0 - mae / 0.2) * 0.2) + // MAE from full predictions (scaled)
                                                  (Math.Abs(simulatedCorrelation) * 0.3) + // Correlation from simulated inference
                                                  (Math.Max(0, 1.0 - simulatedMAE / 0.2) * 0.2); // MAE from simulated inference (scaled)
                                confidenceScore = Math.Round(Math.Max(0, Math.Min(1, confidenceScore)), 2); // Clamp and round
                                summaryParts.Add($"Combined Confidence: {confidenceScore:P0}");
                            }
                            else
                            {
                                summaryParts.Add("Confidence Score N/A");
                            }
                            autoGenOverallSummary = string.Join(" | ", summaryParts);
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: Final Overall Summary: {autoGenOverallSummary}.");

                        }
                        else
                        {
                            Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: Simulated inference produced no valid predictions. Skipping final comparison step.");
                            autoGenOverallSummary = "Simulated Inference Failed - " + autoGenOverallSummary; // Append to existing partial summary
                        }
                    }
                    else
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: Validation data generation failed or mismatch. Skipping simulated inference and final comparison step.");
                        autoGenOverallSummary = "Validation Data Unavailable - " + autoGenOverallSummary; // Append to existing partial summary
                    }
                }
                else
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: Model A and/or Model B predictions were null or empty. Skipping AutoGen collaboration entirely.");
                    autoGenOverallSummary = "Predictions Unavailable - Minimal Analysis";
                }

                // Store AutoGen results and other derived data in the result dictionaries for potential downstream use or logging
                unitAResults["AutoGen_OverallSummary_D"] = autoGenOverallSummary;
                unitBResults["AutoGen_OverallSummary_D"] = autoGenOverallSummary;
                unitAResults["AutoGen_SimulatedOutputA_D"] = simulatedOutputA; // Will be NaN if simulation didn't run
                unitBResults["AutoGen_SimulatedOutputB_D"] = simulatedOutputB; // Will be NaN if simulation didn't run
                // Store the prediction value used for the common input simulation (if applicable)
                unitAResults["AutoGen_SelectedInputPrediction_D"] = selectedPredictionIndex != -1 && modelAPredictions?.Length > selectedPredictionIndex ? modelAPredictions[selectedPredictionIndex] : float.NaN;
                unitBResults["AutoGen_SelectedInputPrediction_D"] = selectedPredictionIndex != -1 && modelBPredictions?.Length > selectedPredictionIndex ? modelBPredictions[selectedPredictionIndex] : float.NaN;


                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Agent Collaboration: AutoGen workflow completed. Overall summary: {autoGenOverallSummary}");

                // Conceptual Model Merging (if parameters are available)
                byte[]? mergedModelData = null;
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: SequentialFinalProcessingUnitD: Attempting conceptual model merge.");

                if (modelACombinedParams != null && modelACombinedParams.Length > 0 && modelBCombinedParams != null && modelBCombinedParams.Length > 0)
                {
                    try
                    {
                        mergedModelData = modelACombinedParams.Concat(modelBCombinedParams).ToArray();
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Conceptually merged Model A ({modelACombinedParams.Length} bytes) and Model B ({modelBCombinedParams.Length} bytes) parameters. Merged data size: {mergedModelData.Length} bytes.");
                    }
                    catch (Exception mergeEx)
                    {
                        Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Error during conceptual model merge: {mergeEx.Message}");
                        mergedModelData = new byte[0]; // Ensure it's initialized on error
                    }
                }
                else if (modelACombinedParams != null && modelACombinedParams.Length > 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Only Model A parameters available. Using Model A parameters ({modelACombinedParams.Length} bytes) as merged data (conceptually).");
                    mergedModelData = modelACombinedParams;
                }
                else if (modelBCombinedParams != null && modelBCombinedParams.Length > 0)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Only Model B parameters available. Using Model B parameters ({modelBCombinedParams.Length} bytes) as merged data (conceptually).");
                    mergedModelData = modelBCombinedParams;
                }
                else
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Neither Model A nor Model B parameters were found. Cannot perform conceptual merge. Merged data is empty.");
                    mergedModelData = new byte[0]; // Ensure it's initialized
                }


                RuntimeProcessingContext.StoreContextValue("merged_model_params", mergedModelData);
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Stored conceptual merged model data ({mergedModelData?.Length ?? 0} bytes) in RuntimeContext.");


                // Final update to the CoreMlOutcomeRecord
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: SequentialFinalProcessingUnitD: Updating CoreMlOutcomeRecord with final details.");
                outcomeRecord.OutcomeGenerationTimestamp = DateTime.UtcNow;

                // Populate derived vectors with information about predictions and simulations
                string modelAPredInfo = modelAPredictions != null && modelAPredictions.Length > 0
                    ? $"ModelA_Preds_Count:{modelAPredictions.Length}_BestMatchIdx:{selectedPredictionIndex}_InputUsed:{(selectedPredictionIndex != -1 && selectedPredictionIndex < modelAPredictions.Length ? modelAPredictions[selectedPredictionIndex].ToString("F6") : "N/A")}"
                    : "ModelA_Preds_Unavailable";
                outcomeRecord.DerivedProductFeatureVector = modelAPredInfo;

                string modelBPredSimInfo = modelBPredictions != null && modelBPredictions.Length > 0
                    ? $"ModelB_Preds_Count:{modelBPredictions.Length}_SimOutputA:{simulatedOutputA:F6}_SimOutputB:{simulatedOutputB:F6}"
                    : "ModelB_Preds_Unavailable";
                outcomeRecord.DerivedServiceBenefitVector = modelBPredSimInfo;


                // Update classification based on data availability and processing status
                int classificationId;
                string classificationDescription = outcomeRecord.CategoricalClassificationDescription ?? "";
                // Clean up previous status indicators before adding new ones
                classificationDescription = classificationDescription.Replace(" (Unit A Warn)", "").Replace(" (Unit B Warn)", "").Replace(" (Unit A Error)", "").Replace(" (Unit B Error)", "").Replace(" (TrainingError)", "");

                // Check data availability for classification
                bool partialDataAvailable = !fullDataAvailable && ((modelACombinedParams?.Length ?? 0) > 0 || (modelBCombinedParams?.Length ?? 0) > 0 || (modelAPredictions?.Length ?? 0) > 0 || (modelBPredictions?.Length ?? 0) > 0);


                if (fullDataAvailable)
                {
                    classificationId = 250; // Indicate full processing
                    classificationDescription += $" (Full Data Processed, Analysis: {autoGenOverallSummary})";
                }
                else if (partialDataAvailable)
                {
                    classificationId = 150; // Indicate partial processing
                    classificationDescription += $" (Partial Data Processed, Analysis: {autoGenOverallSummary})";
                }
                else
                {
                    classificationId = 50; // Indicate minimal/no data for processing
                    classificationDescription += $" (No Model Data, Analysis: {autoGenOverallSummary})";
                }

                // Append error/warning flags from individual units
                if (unitAResults.ContainsKey("ModelAProcessingError")) classificationDescription += " (Unit A Error)";
                else if (unitAResults.ContainsKey("ModelAProcessingWarning")) classificationDescription += " (Unit A Warn)";

                if (unitBResults.ContainsKey("ModelBProcessingError")) classificationDescription += " (Unit B Error)";
                else if (unitBResults.ContainsKey("ModelBProcessingWarning")) classificationDescription += " (Unit B Warn)";

                outcomeRecord.CategoricalClassificationIdentifier = classificationId;
                outcomeRecord.CategoricalClassificationDescription = classificationDescription;


                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Final Outcome Record Details:\n  - RecordIdentifier: {outcomeRecord.RecordIdentifier}\n  - AssociatedCustomerIdentifier: {outcomeRecord.AssociatedCustomerIdentifier}\n  - OutcomeGenerationTimestamp: {outcomeRecord.OutcomeGenerationTimestamp}\n  - CategoricalClassificationIdentifier: {outcomeRecord.CategoricalClassificationIdentifier}\n  - CategoricalClassificationDescription: {outcomeRecord.CategoricalClassificationDescription}\n  - SerializedSimulatedModelData Size: {outcomeRecord.SerializedSimulatedModelData?.Length ?? 0} bytes\n  - AncillaryBinaryDataPayload Size: {outcomeRecord.AncillaryBinaryDataPayload?.Length ?? 0} bytes\n  - DerivedProductFeatureVector: {outcomeRecord.DerivedProductFeatureVector}\n  - DerivedServiceBenefitVector: {outcomeRecord.DerivedServiceBenefitVector}");

                // Save the final state of the outcome record to simulated persistence
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: SequentialFinalProcessingUnitD: Attempting to save final CoreMlOutcomeRecord to simulated persistence.");
                var recordIndex = InMemoryTestDataSet.SimulatedCoreOutcomes.FindIndex(r => r.RecordIdentifier == outcomeRecord.RecordIdentifier);
                if (recordIndex >= 0)
                {
                    lock (InMemoryTestDataSet.SimulatedCoreOutcomes) // Ensure thread-safe update
                    {
                        InMemoryTestDataSet.SimulatedCoreOutcomes[recordIndex] = outcomeRecord;
                    }
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Final CoreMlOutcomeRecord (ID: {outcomeRecord.RecordIdentifier}) state saved successfully to simulated persistent storage.");
                }
                else
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Error - CoreMlOutcomeRecord with Identifier {outcomeRecord.RecordIdentifier} not found in simulated storage during final update attempt!");
                }

                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Sequential Final Processing Unit D (Actual Model D Concept with AutoGen) completed all processing steps successfully.");
            }
            catch (Exception ex) // Inner try-catch for the main logic
            {
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Unhandled Error in Sequential Final Processing Unit D: {ex.Message}");
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Stack Trace: {ex.StackTrace}");

                // Update outcome record with error state
                outcomeRecord.CategoricalClassificationDescription = (outcomeRecord.CategoricalClassificationDescription ?? "") + " (FinalProcessingError)";
                outcomeRecord.OutcomeGenerationTimestamp = DateTime.UtcNow;
                outcomeRecord.CategoricalClassificationIdentifier = Math.Min(outcomeRecord.CategoricalClassificationIdentifier ?? 50, 100); // Use a lower ID for error

                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: SequentialFinalProcessingUnitD: Attempting to save CoreMlOutcomeRecord with error state.");
                var recordIndex = InMemoryTestDataSet.SimulatedCoreOutcomes.FindIndex(r => r.RecordIdentifier == outcomeRecord.RecordIdentifier);
                if (recordIndex >= 0)
                {
                    lock (InMemoryTestDataSet.SimulatedCoreOutcomes)
                    {
                        InMemoryTestDataSet.SimulatedCoreOutcomes[recordIndex] = outcomeRecord;
                    }
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Updated simulated persistent storage with error state.");
                }
                else
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Error state occurred, but CoreMlOutcomeRecord with Identifier {outcomeRecord.RecordIdentifier} was not found for saving.");
                }
                throw; // Re-throw to be caught by the orchestrator
            }
            finally // Corresponds to the outer try
            {
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Workflow Session {requestSequenceIdentifier}: Sequential Final Processing Unit D (Actual Model D Concept with AutoGen) finished execution.");
            }
        }


        // Helper method to serialize a float array to a byte array
        private byte[] SerializeFloatArray(float[] data)
        {
            if (data == null) return new byte[0];
            var byteList = new List<byte>();
            foreach (var f in data)
            {
                byteList.AddRange(BitConverter.GetBytes(f));
            }
            return byteList.ToArray();
        }

        // Helper method to deserialize a byte array back to a float array
        private float[] DeserializeFloatArray(byte[] data)
        {
            if (data == null || data.Length == 0) return new float[0];
            if (data.Length % 4 != 0) // Size of float is 4 bytes
            {
                Console.WriteLine($"Warning: Byte array length ({data.Length}) is not a multiple of 4 for deserialization.");
                return new float[0];
            }
            var floatArray = new float[data.Length / 4];
            // Specify System.Buffer
            System.Buffer.BlockCopy(data, 0, floatArray, 0, data.Length);
            return floatArray;
        }


        // Adapted from GET endpoints - methods to retrieve records from simulated persistence.
        public CoreMlOutcomeRecord? GetOutcomeRecordByIdentifier(int recordIdentifier)
        {
            // Operational Step: Retrieve a single record from simulated persistence by its unique identifier.
            // Operational Process Dependency: Reads from InMemoryTestDataSet.SimulatedCoreOutcomes.
            // Subsequent Usage: Returns the found record or null.
            lock (InMemoryTestDataSet.SimulatedCoreOutcomes) // Added lock for thread safety
            {
                var outcomeRecord = InMemoryTestDataSet.SimulatedCoreOutcomes.FirstOrDefault(r => r.RecordIdentifier == recordIdentifier);
                if (outcomeRecord == null)
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Request for record ID {recordIdentifier}: Record not found.");
                }
                else
                {
                    Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Request for record ID {recordIdentifier}: Record found.");
                }
                return outcomeRecord;
            }
        }

        public IEnumerable<CoreMlOutcomeRecord> GetAllOutcomeRecords()
        {
            // Operational Step: Retrieve all records from simulated persistence.
            // Operational Process Dependency: Reads from InMemoryTestDataSet.SimulatedCoreOutcomes.
            // Subsequent Usage: Returns the list of all records currently in the simulated storage.
            lock (InMemoryTestDataSet.SimulatedCoreOutcomes) // Added lock for thread safety
            {
                Console.WriteLine($"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] Request for all records: Returning {InMemoryTestDataSet.SimulatedCoreOutcomes.Count} records.");
                return InMemoryTestDataSet.SimulatedCoreOutcomes.ToList(); // Return a copy to avoid issues if the list is modified later
            }
        }
    }

    // Extension method to reshape a 1D array into a multi-dimensional array
    // Note: TensorFlow.NET's Tensor.reshape() is for Tensor objects, not .NET arrays.
    // This helper is needed to convert the deserialized float[] into the shape expected by tf.constant().
    public static class ArrayExtensions
    {
        public static Array reshape<T>(this T[] flatArray, params int[] dimensions)
        {
            if (flatArray == null) throw new ArgumentNullException(nameof(flatArray));
            if (dimensions == null) throw new ArgumentNullException(nameof(dimensions));

            int totalSize = 1;
            checked // Use checked to catch overflow if dimensions are too large
            {
                foreach (var dim in dimensions)
                {
                    if (dim < 0) throw new ArgumentException("Dimensions cannot be negative.", nameof(dimensions));
                    totalSize *= dim;
                }
            }

            if (totalSize != flatArray.Length)
            {
                throw new ArgumentException($"Total size of new dimensions ({totalSize}) must match the size of the array ({flatArray.Length}).", nameof(dimensions));
            }

            // For simplicity, this helper only supports 1D and 2D reshaping to [N, M].
            if (dimensions.Length == 1)
            {
                // Reshaping to 1D just returns the original array
                return flatArray;
            }
            else if (dimensions.Length == 2)
            {
                int rows = dimensions[0];
                int cols = dimensions[1];
                T[,] reshaped = new T[rows, cols];
                // Using BlockCopy is efficient for primitive types, but requires a primitive array type T
                if (typeof(T).IsPrimitive || typeof(T) == typeof(float) || typeof(T) == typeof(double)) // Explicitly check for common types
                {
                    System.Buffer.BlockCopy(flatArray, 0, reshaped, 0, System.Buffer.ByteLength(flatArray));
                }
                else
                {
                    // Fallback for non-primitive types - slower element by element copy
                    int flatIndex = 0;
                    for (int i = 0; i < rows; i++)
                    {
                        for (int j = 0; j < cols; j++)
                        {
                            reshaped[i, j] = flatArray[flatIndex++];
                        }
                    }
                }
                return reshaped;
            }
            else
            {
                // Throw for unsupported shapes
                throw new NotSupportedException($"Reshaping to {string.Join(",", dimensions)} is not supported by this helper. Only 1D and 2D arrays are supported.");
            }
        }
    }


    internal class Program
    {
        static async Task Main(string[] args) // Make Main async to call the async orchestrator method
        {
            Console.WriteLine("Hello, Agentic ML World!");
            Console.WriteLine("Initializing ML Process Orchestrator...");

            var orchestrator = new MlProcessOrchestrator();

            Console.WriteLine("Enter customer identifier to initiate ML process (or type 'list' or 'get [id]'):");
            string? input = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(input))
            {
                Console.WriteLine("No input provided. Exiting.");
                return;
            }

            if (input.Trim().Equals("list", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine("Fetching all outcome records...");
                try
                {
                    var allRecords = orchestrator.GetAllOutcomeRecords().ToList();
                    if (allRecords.Any())
                    {
                        Console.WriteLine($"Found {allRecords.Count} records:");
                        foreach (var record in allRecords)
                        {
                            Console.WriteLine($"- ID: {record.RecordIdentifier}, Customer: {record.AssociatedCustomerIdentifier}, Timestamp: {record.OutcomeGenerationTimestamp}, Class: {record.CategoricalClassificationDescription}");
                        }
                    }
                    else
                    {
                        Console.WriteLine("No records found in memory.");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error fetching records: {ex.Message}");
                }
                Console.WriteLine("Press Enter to exit.");
                Console.ReadLine();
                return;
            }
            else if (input.Trim().StartsWith("get ", StringComparison.OrdinalIgnoreCase))
            {
                var parts = input.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length == 2 && int.TryParse(parts[1], out int recordId))
                {
                    Console.WriteLine($"Fetching record with ID: {recordId}...");
                    try
                    {
                        var record = orchestrator.GetOutcomeRecordByIdentifier(recordId);
                        if (record != null)
                        {
                            Console.WriteLine($"Record Found:");
                            Console.WriteLine(Newtonsoft.Json.JsonConvert.SerializeObject(record, Newtonsoft.Json.Formatting.Indented)); // Use Newtonsoft.Json for pretty printing
                        }
                        else
                        {
                            Console.WriteLine($"Record with ID {recordId} not found.");
                        }
                    }
                    catch (FileNotFoundException fnfEx)
                    {
                        Console.WriteLine($"Error fetching record: {fnfEx.Message}"); // Handle specific errors if GetOutcomeRecordByIdentifier changes
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error fetching record: {ex.Message}");
                    }
                    Console.WriteLine("Press Enter to exit.");
                    Console.ReadLine();
                    return;
                }
                else
                {
                    Console.WriteLine("Invalid 'get' command format. Use 'get [recordId]'.");
                    Console.WriteLine("Press Enter to exit.");
                    Console.ReadLine();
                    return;
                }
            }
            else if (int.TryParse(input, out int customerId))
            {
                Console.WriteLine($"Initiating ML process for customer {customerId}...");

                try
                {
                    // Call the core orchestration logic
                    CoreMlOutcomeRecord finalResult = await orchestrator.InitiateMlOutcomeGeneration(customerId);

                    Console.WriteLine("\n--- ML Process Completed ---");
                    Console.WriteLine($"Final Outcome for Customer {customerId}:");
                    Console.WriteLine($"Record ID: {finalResult.RecordIdentifier}");
                    Console.WriteLine($"Timestamp: {finalResult.OutcomeGenerationTimestamp}");
                    Console.WriteLine($"Classification ID: {finalResult.CategoricalClassificationIdentifier}");
                    Console.WriteLine($"Classification Description: {finalResult.CategoricalClassificationDescription}");
                    Console.WriteLine($"Product Vector Info: {finalResult.DerivedProductFeatureVector}");
                    Console.WriteLine($"Service Vector Info: {finalResult.DerivedServiceBenefitVector}");
                    Console.WriteLine($"Serialized Model Data Size: {finalResult.SerializedSimulatedModelData?.Length ?? 0} bytes");
                    Console.WriteLine($"Ancillary Data Payload Size: {finalResult.AncillaryBinaryDataPayload?.Length ?? 0} bytes");


                    // Also retrieve and print AutoGen summary if available
                    var agSummaryA = RuntimeProcessingContext.RetrieveContextValue("AutoGen_OverallSummary_D");
                    if (agSummaryA != null)
                    {
                        Console.WriteLine($"AutoGen Analysis Summary (Unit D): {agSummaryA}");
                    }
                    var agSimA = RuntimeProcessingContext.RetrieveContextValue("AutoGen_SimulatedOutputA_D");
                    var agSimB = RuntimeProcessingContext.RetrieveContextValue("AutoGen_SimulatedOutputB_D");
                    if (agSimA is float simAFloat && agSimB is float simBFloat && !float.IsNaN(simAFloat) && !float.IsNaN(simBFloat))
                    {
                        Console.WriteLine($"Simulated Final Output Comparison (Unit D): Model A = {simAFloat:F6}, Model B = {simBFloat:F6}");
                    }


                }
                catch (ArgumentOutOfRangeException argEx)
                {
                    Console.Error.WriteLine($"Input Error: {argEx.Message}");
                }
                catch (InvalidOperationException ioEx)
                {
                    Console.Error.WriteLine($"Workflow Error: {ioEx.Message}");
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"An unexpected error occurred: {ex.Message}");
                    Console.Error.WriteLine(ex.StackTrace); // Print stack trace for unhandled errors
                }
            }
            else
            {
                Console.WriteLine("Invalid input. Please enter a valid customer identifier or command ('list', 'get [id]').");
            }

            Console.WriteLine("\nPress Enter to exit.");
            Console.ReadLine();
        }
    }
}

